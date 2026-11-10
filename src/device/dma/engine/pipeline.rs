//! The BD pipeline of a stalled channel: which BDs the channel has loaded, what `Cur_BD` names,
//! how many tasks it holds before `Task_Queue_Size` counts, and what the `Status[1:0]` field reads.
//!
//! Everything here is measured (Phoenix NPU1, shim-queue probes, tools/gen_dma_queue_probe.py:
//! the `stall`, `absorb` and ramp families; every number below was bisected to the word or BD).
//! The channel keeps its stalled BD and the BDs behind it (the rest of the task's chain, then the
//! first BDs of the queued tasks) loaded in a short pipeline, and the status register shows it:
//!
//! - `Cur_BD` is the last BD the pipeline has taken, up to `last_visible` BDs past the stalled one;
//! - `Status[1:0]` (the head counted by the words it still has to move) is 1 (STARTING) when the pipeline is full by count, otherwise 2 (RUNNING) when
//!   the loaded BDs together move at least a threshold number of words, otherwise 1 when the next
//!   BD did not fit the pipeline's word budget, otherwise 0;
//! - a BD joins the pipeline only while the loaded BDs stay within a word budget (the BDs behind
//!   the head also cost a few words each);
//! - a push into a queue that is full behind a full pipeline blocks the writer.
//!
//! The pipeline is deeper when the head is stalled on its data than when it is stalled on a lock
//! acquire, and deeper on a shim than on a compute tile or memtile.

use super::*;

/// What stalls the head BD of a channel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum HeadStall {
    /// Parked on a lock acquire.
    Lock,
    /// Past its lock, working on (or waiting for) the stream: starved S2MM, backpressured MM2S.
    Data,
}

/// BDs a pipeline holds, and how many of them `Cur_BD` can name past the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PipelineGeometry {
    /// BDs loaded when the pipeline is full, the head included.
    pub depth: usize,
    /// Index (0 = the head) of the last BD `Cur_BD` can name.
    pub last_visible: usize,
    /// Words the loaded BDs may move together, counting `BD_OVERHEAD_WORDS` for every BD after
    /// the head (silicon, bisected: chains of equal BDs, the last length that still loaded its
    /// second or third BD, on every tile kind; the overhead and the budget were fitted jointly,
    /// so each budget is a range of two or three words: 38..41, 28..29, 206..208, 136..137,
    /// and the middle or low end is used).
    pub budget: u32,
}

/// What every BD behind the head costs against the pipeline's budget, besides its words.
const BD_OVERHEAD_WORDS: u32 = 4;

/// One BD in a channel's pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LoadedBd {
    pub bd: u8,
    pub words: u32,
}

impl DmaEngine {
    /// The stall of the channel's head BD, and the direction it moves data in.
    pub(super) fn head_stall(&self, channel: u8) -> Option<(HeadStall, TransferDirection)> {
        let ch = self.channels.get(channel as usize)?;
        match &ch.fsm {
            ChannelFsm::AcquiringLock { acquired: false, transfer, .. } => {
                Some((HeadStall::Lock, transfer.direction))
            }
            ChannelFsm::Idle | ChannelFsm::Error => None,
            other => other.transfer().map(|t| (HeadStall::Data, t.direction)),
        }
    }

    /// Pipeline depth and `Cur_BD` reach (silicon: parked on a lock acquire, every tile kind:
    /// 3 and 2; stream-stalled, compute tile and memtile: 5 and 4; shim: 7 and 5, whose last
    /// loaded BD is not named). The MM2S backpressure case is probed only through chains of
    /// 2-word BDs, whose `Cur_BD` is consistent with the S2MM numbers.
    pub(super) fn pipeline_geometry(&self, stall: HeadStall) -> PipelineGeometry {
        match (stall, self.tile_kind.is_shim()) {
            (HeadStall::Lock, false) => PipelineGeometry { depth: 3, last_visible: 2, budget: 40 },
            (HeadStall::Lock, true) => PipelineGeometry { depth: 3, last_visible: 2, budget: 207 },
            (HeadStall::Data, false) => PipelineGeometry { depth: 5, last_visible: 4, budget: 28 },
            (HeadStall::Data, true) => PipelineGeometry { depth: 7, last_visible: 5, budget: 136 },
        }
    }

    /// Words the head BD and the one behind it must move together for `Status` to read RUNNING
    /// (silicon, bisected: a single BD of 21 words parked on a lock reads 2 and 20 reads 0 on
    /// a compute tile or memtile, 137 and 136 on a shim S2MM; stream-stalled S2MM 15 and 69; a
    /// backpressured MM2S 57 on a compute tile or memtile). A shim MM2S never reads RUNNING:
    /// nothing up to 512 words did.
    fn running_threshold_words(&self, stall: HeadStall, direction: TransferDirection) -> Option<u32> {
        use TransferDirection::{MM2S, S2MM};
        let shim = self.tile_kind.is_shim();
        match (stall, direction, shim) {
            (HeadStall::Lock, S2MM, true) => Some(137),
            (HeadStall::Lock, MM2S, true) => None,
            (HeadStall::Lock, _, false) => Some(21),
            (HeadStall::Data, S2MM, true) => Some(69),
            (HeadStall::Data, S2MM, false) => Some(15),
            (HeadStall::Data, MM2S, true) => None,
            (HeadStall::Data, MM2S, false) => Some(57),
        }
    }

    /// The BDs the channel has loaded, head first: the rest of the head's chain, then the first
    /// BDs of the queued tasks, while they fit the pipeline's depth and word budget. The flag says
    /// that a further BD exists but did not fit the budget.
    pub(super) fn loaded_bds(&self, channel: u8, stall: HeadStall) -> (Vec<LoadedBd>, bool) {
        let Some(ch) = self.channels.get(channel as usize) else {
            return (Vec::new(), false);
        };
        let Some(transfer) = ch.fsm.transfer() else {
            return (Vec::new(), false);
        };
        let head = transfer.bd_index;
        let geometry = self.pipeline_geometry(stall);
        let words = |bd: u8| self.bd_configs.get(bd as usize).map_or(0, |c| c.length / 4);
        // the BDs that would follow, in order: the chain, then the queued tasks
        let mut behind = Vec::new();
        let mut bd = head;
        while let Some(next) = self.bd_configs.get(bd as usize).and_then(|c| c.next_bd) {
            if next == head || behind.iter().any(|l: &LoadedBd| l.bd == next) {
                break;
            }
            behind.push(LoadedBd { bd: next, words: words(next) });
            bd = next;
        }
        behind.extend(
            ch.task_queue
                .iter()
                .map(|t| LoadedBd { bd: t.start_bd, words: words(t.start_bd) }),
        );

        // what the head still has to move: a starved S2MM that got part of its BD reads by the rest
        // (silicon, lock-leftovers: 112 words of which 32 arrived read RUNNING, 96 and 80 did not)
        let mut out = vec![LoadedBd { bd: head, words: (transfer.remaining_bytes() / 4) as u32 }];
        let mut spent = out[0].words;
        for next in behind {
            if out.len() >= geometry.depth {
                return (out, false);
            }
            let cost = next.words + BD_OVERHEAD_WORDS;
            if spent + cost > geometry.budget {
                return (out, true);
            }
            spent += cost;
            out.push(next);
        }
        (out, false)
    }

    /// The BD `Cur_BD` names while the head is stalled and BDs sit behind it: the last one the
    /// pipeline has taken, within its reach. `None` when nothing sits behind the head.
    pub(super) fn pipeline_cur_bd(&self, channel: u8) -> Option<u8> {
        let (stall, _) = self.head_stall(channel)?;
        let (loaded, _) = self.loaded_bds(channel, stall);
        if loaded.len() < 2 {
            return None;
        }
        let last = self.pipeline_geometry(stall).last_visible.min(loaded.len() - 1);
        Some(loaded[last].bd)
    }

    /// The `Status[1:0]` field of the channel (0 IDLE, 1 STARTING, 2 RUNNING).
    pub(super) fn status_field(&self, channel: u8) -> u32 {
        let Some((stall, direction)) = self.head_stall(channel) else {
            return 0;
        };
        let (loaded, budget_limited) = self.loaded_bds(channel, stall);
        let mut moved = loaded.iter().map(|l| l.words).sum::<u32>();
        // A backpressured staged MM2S counts its head BD whole: its fetched words have left the
        // transfer (they wait in the channel's staging and the route) but silicon still reads a
        // single 57-word BD stalled on its stream as RUNNING whatever it has fetched (shim-queue
        // stall probes, BDs of 57, 60 and 64 words into a consumer that never starts).
        if let Some(transfer) = self.channels[channel as usize].fsm.transfer() {
            if stall == HeadStall::Data && self.uses_egress_staging(transfer) {
                moved += ((transfer.total_bytes - transfer.remaining_bytes()) / 4) as u32;
            }
        }
        if loaded.len() >= self.pipeline_geometry(stall).depth {
            1
        } else if self.running_threshold_words(stall, direction).is_some_and(|t| moved >= t) {
            2
        } else if budget_limited {
            1
        } else {
            0
        }
    }

    /// Whether the channel's pipeline is full of tasks: the next push counts in
    /// `Task_Queue_Size` (and the one after the queue's four blocks the writer).
    pub fn staging_full(&self, channel: u8) -> bool {
        self.pending_tasks(channel) > 0 && self.pending_tasks(channel) >= self.staged_task_slots(channel)
    }
}
