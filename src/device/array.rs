//! AIE tile array representation.
//!
//! The tile array is the top-level device state. It holds all tiles
//! and provides methods to access them efficiently.
//!
//! # NPU Array Layouts
//!
//! | Device | Columns | Rows | Layout |
//! |--------|---------|------|--------|
//! | NPU1 | 5 | 6 | Col 0 shim, Rows 0 shim, 1 mem, 2-5 compute |
//! | NPU2 | 5 | 6 | Same as NPU1 |
//! | NPU3 | 9 | 6 | 8 columns + shim |
//!
//! # Performance
//!
//! Tiles are stored in a flat Vec for cache efficiency. Access is O(1)
//! via `col * rows + row` indexing.
//!
//! # DMA Integration
//!
//! Each tile has an associated `DmaEngine` stored in a parallel array.
//! DMA engines are accessed via `dma_engine(col, row)` and stepped via
//! `step_dma(col, row, host_memory)`.

use super::dma::{DmaEngine, DmaResult};
use super::host_memory::HostMemory;
use super::stream_router::StreamRouter;
use super::tile::{Tile, TileType};
use super::AieArch;

/// Maximum supported columns
pub const MAX_COLS: usize = 9;

/// Maximum supported rows
pub const MAX_ROWS: usize = 6;

/// AIE tile array.
///
/// Stores tiles in row-major order for cache-friendly column iteration.
pub struct TileArray {
    /// Architecture variant
    arch: AieArch,

    /// Number of columns (including shim column)
    cols: u8,

    /// Number of rows
    rows: u8,

    /// Tiles stored in flat array: tiles[col * rows + row]
    /// Using Vec because tile count varies by device
    pub(crate) tiles: Vec<Tile>,

    /// DMA engines stored in parallel with tiles.
    /// Each tile has exactly one DMA engine.
    pub(crate) dma_engines: Vec<DmaEngine>,

    /// Stream router for tile-to-tile data flow.
    pub stream_router: StreamRouter,
}

impl TileArray {
    /// Create a new tile array for the given architecture.
    pub fn new(arch: AieArch) -> Self {
        let cols = arch.columns();
        let rows = arch.rows();
        let capacity = (cols as usize) * (rows as usize);

        let mut tiles = Vec::with_capacity(capacity);
        let mut dma_engines = Vec::with_capacity(capacity);

        // Create tiles and DMA engines in column-major order
        for col in 0..cols {
            for row in 0..rows {
                let tile_type = if arch.is_shim_tile(col, row) {
                    TileType::Shim
                } else if arch.is_mem_tile(col, row) {
                    TileType::MemTile
                } else {
                    TileType::Compute
                };
                tiles.push(Tile::new(tile_type, col, row));

                // Create appropriate DMA engine for tile type
                let engine = match tile_type {
                    TileType::MemTile => DmaEngine::new_mem_tile(col, row),
                    _ => DmaEngine::new_compute_tile(col, row),
                };
                dma_engines.push(engine);
            }
        }

        let stream_router = StreamRouter::new(cols as usize, rows as usize);

        Self { arch, cols, rows, tiles, dma_engines, stream_router }
    }

    /// Create an NPU1 (Phoenix/HawkPoint) array.
    #[inline]
    pub fn npu1() -> Self {
        Self::new(AieArch::Aie2)
    }

    /// Create an NPU2 (Strix) array.
    #[inline]
    pub fn npu2() -> Self {
        Self::new(AieArch::Aie2P)
    }

    /// Get the architecture.
    #[inline]
    pub fn arch(&self) -> AieArch {
        self.arch
    }

    /// Get number of columns.
    #[inline]
    pub fn cols(&self) -> u8 {
        self.cols
    }

    /// Get number of rows.
    #[inline]
    pub fn rows(&self) -> u8 {
        self.rows
    }

    /// Get tile index from coordinates.
    #[inline]
    fn tile_index(&self, col: u8, row: u8) -> usize {
        (col as usize) * (self.rows as usize) + (row as usize)
    }

    /// Get a tile by coordinates.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn get(&self, col: u8, row: u8) -> Option<&Tile> {
        if col < self.cols && row < self.rows {
            Some(&self.tiles[self.tile_index(col, row)])
        } else {
            None
        }
    }

    /// Get a mutable tile by coordinates.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn get_mut(&mut self, col: u8, row: u8) -> Option<&mut Tile> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            Some(&mut self.tiles[idx])
        } else {
            None
        }
    }

    /// Get a tile by coordinates (panics if out of bounds).
    ///
    /// Use this in hot paths where bounds are known valid.
    #[inline]
    pub fn tile(&self, col: u8, row: u8) -> &Tile {
        debug_assert!(col < self.cols && row < self.rows);
        &self.tiles[self.tile_index(col, row)]
    }

    /// Get a mutable tile by coordinates (panics if out of bounds).
    #[inline]
    pub fn tile_mut(&mut self, col: u8, row: u8) -> &mut Tile {
        debug_assert!(col < self.cols && row < self.rows);
        let idx = self.tile_index(col, row);
        &mut self.tiles[idx]
    }

    /// Iterate over all tiles.
    pub fn iter(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter()
    }

    /// Iterate over all tiles mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tile> {
        self.tiles.iter_mut()
    }

    /// Iterate over compute tiles only.
    pub fn compute_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_compute())
    }

    /// Iterate over compute tiles mutably.
    pub fn compute_tiles_mut(&mut self) -> impl Iterator<Item = &mut Tile> {
        self.tiles.iter_mut().filter(|t| t.is_compute())
    }

    /// Get all shim tiles.
    pub fn shim_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_shim())
    }

    /// Get all memory tiles.
    pub fn mem_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_mem_tile())
    }

    // === DMA Engine Access ===

    /// Get the DMA engine for a tile.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn dma_engine(&self, col: u8, row: u8) -> Option<&DmaEngine> {
        if col < self.cols && row < self.rows {
            Some(&self.dma_engines[self.tile_index(col, row)])
        } else {
            None
        }
    }

    /// Get the mutable DMA engine for a tile.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn dma_engine_mut(&mut self, col: u8, row: u8) -> Option<&mut DmaEngine> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            Some(&mut self.dma_engines[idx])
        } else {
            None
        }
    }

    /// Get tile and DMA engine together (for operations that need both).
    ///
    /// Returns separate references to allow independent mutation.
    pub fn tile_and_dma(&mut self, col: u8, row: u8) -> Option<(&mut Tile, &mut DmaEngine)> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            // Safety: We're returning references to different arrays
            Some((&mut self.tiles[idx], &mut self.dma_engines[idx]))
        } else {
            None
        }
    }

    /// Step the DMA engine for a specific tile.
    ///
    /// This advances the DMA transfer state by one cycle.
    pub fn step_dma(&mut self, col: u8, row: u8, host_memory: &mut HostMemory) -> Option<DmaResult> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            let (tile, engine) = (&mut self.tiles[idx], &mut self.dma_engines[idx]);
            Some(engine.step(tile, host_memory))
        } else {
            None
        }
    }

    /// Step all DMA engines.
    ///
    /// Returns true if any DMA engine is still active.
    pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
        let mut any_active = false;
        for i in 0..self.tiles.len() {
            let result = self.dma_engines[i].step(&mut self.tiles[i], host_memory);
            if matches!(result, DmaResult::InProgress | DmaResult::WaitingForLock(_)) {
                any_active = true;
            }
        }
        any_active
    }

    /// Check if any DMA engine has active transfers.
    pub fn any_dma_active(&self) -> bool {
        self.dma_engines.iter().any(|e| e.any_channel_active())
    }

    /// Step the stream router.
    ///
    /// Routes data between tiles through configured stream switch paths.
    /// Returns true if any data was moved.
    pub fn step_streams(&mut self) -> bool {
        self.stream_router.step()
    }

    /// Step all per-tile stream switches.
    ///
    /// This handles intra-tile routing: forwarding data from input (slave) ports
    /// to output (master) ports within each tile based on configured local routes.
    /// This is necessary for pass-through routing where data enters a tile from
    /// one direction and exits to another.
    ///
    /// Returns the total number of words forwarded across all tiles.
    pub fn step_tile_switches(&mut self) -> usize {
        let mut total_forwarded = 0;
        for tile in &mut self.tiles {
            total_forwarded += tile.stream_switch.step();
        }
        total_forwarded
    }

    /// Route DMA MM2S output to stream router.
    ///
    /// This is phase 1 of the DMA-stream routing: moving data from DMA output
    /// to the global stream fabric.
    ///
    /// Returns the number of words routed.
    pub fn route_dma_to_stream(&mut self) -> usize {
        use super::stream_router::StreamWord;

        let mut words_routed = 0;

        // Move data from DMA stream_out to StreamRouter output FIFOs
        // Skip tiles with local routes - they use route_dma_to_tile_switches instead
        for i in 0..self.dma_engines.len() {
            // Skip tiles that have local stream switch routes - these use the TileSwitch path
            if self.tiles[i].stream_switch.local_route_count() > 0 {
                continue;
            }

            let col = (i / self.rows as usize) as u8;
            let row = (i % self.rows as usize) as u8;

            // Check stream_out length before popping
            let stream_out_len = self.dma_engines[i].stream_out_len();
            if stream_out_len > 0 && (col == 0 && row == 0) {
                log::info!("route_dma_to_stream: DMA({},{}) has {} items in stream_out", col, row, stream_out_len);
            }

            while let Some(data) = self.dma_engines[i].pop_stream_out() {
                // Map DMA channel to stream port
                let port = data.channel;
                let stream_word = StreamWord {
                    data: data.data,
                    tlast: data.tlast,
                };
                log::info!("route_dma_to_stream: routing ({},{}) port {} data=0x{:08X} to router", col, row, port, data.data);
                if self.stream_router.write_output(col, row, port, stream_word) {
                    words_routed += 1;
                } else {
                    log::warn!("route_dma_to_stream: write_output failed for ({},{}) port {}", col, row, port);
                    // Backpressure - put data back (can't easily do this, so we accept loss)
                    // In a real system, DMA would stall
                    break;
                }
            }
        }

        words_routed
    }

    /// Route stream router input to DMA S2MM.
    ///
    /// This is phase 2 of the DMA-stream routing: moving data from the stream
    /// fabric to DMA input. This MUST be called AFTER step_streams() so that
    /// data has been routed from source to destination tiles.
    ///
    /// Returns the number of words routed.
    pub fn route_stream_to_dma(&mut self) -> usize {
        let mut words_routed = 0;

        // Move data from StreamRouter input FIFOs to DMA stream_in
        for i in 0..self.dma_engines.len() {
            let col = (i / self.rows as usize) as u8;
            let row = (i % self.rows as usize) as u8;

            // Check if this DMA engine needs data for S2MM
            let needs_data = self.dma_engines[i].s2mm_needs_data();
            let has_router_data = self.stream_router.input_has_data(col, row, 0);

            // Debug: log state for compute tile (0,2)
            if col == 0 && row == 2 && (needs_data.is_some() || has_router_data) {
                let ch0_state = self.dma_engines[i].channel_state(0);
                let stream_in_len = self.dma_engines[i].stream_in_len();
                log::info!(
                    "route_stream_to_dma: tile ({},{}) needs_data={:?}, ch0_state={:?}, stream_in_len={}, router_has_data={}",
                    col, row, needs_data, ch0_state, stream_in_len, has_router_data
                );
            }

            if let Some(channel) = needs_data {
                use super::aie2_spec::stream_switch::shim;
                log::trace!("DMA ({},{}) S2MM channel {} needs data", col, row, channel);
                // Try to get data from the stream router
                // Port mapping depends on tile type:
                // - Shim: S2MM receives from North ports (per aie2_spec::stream_switch::shim)
                //   Global routes deliver to the slave port index, so we check North ports.
                // - Other tiles: S2MM receives on DMA ports (port = channel typically)
                let ports_to_check: Vec<u8> = if self.arch.is_shim_tile(col, row) {
                    // Shim: check North slave ports for incoming data
                    (shim::NORTH_SLAVE_START..=shim::NORTH_SLAVE_END).collect()
                } else {
                    // Compute/MemTile: check DMA port
                    vec![if channel == 0 { 0 } else if channel == 1 { 1 } else { channel }]
                };

                let mut found_data = false;
                for port in ports_to_check {
                    if found_data {
                        break; // Only route one word per step
                    }
                    if let Some(stream_word) = self.stream_router.read_input(col, row, port) {
                        let stream_data = super::dma::StreamData {
                            data: stream_word.data,
                            tlast: stream_word.tlast,
                            channel,
                        };
                        if self.dma_engines[i].push_stream_in(stream_data) {
                            log::info!("  -> routed word 0x{:08X} from port {} to DMA ch {}", stream_word.data, port, channel);
                            words_routed += 1;
                            found_data = true;
                        } else {
                            log::warn!("  -> push_stream_in failed for DMA ({},{})", col, row);
                        }
                    }
                }
            }
        }

        // Also move data to tile stream input buffers for direct core access
        for i in 0..self.tiles.len() {
            let col = (i / self.rows as usize) as u8;
            let row = (i % self.rows as usize) as u8;

            // Check all stream input ports (0-7) for data
            for port in 0..8u8 {
                // Only route if there's data available
                if self.stream_router.input_has_data(col, row, port) {
                    if let Some(stream_word) = self.stream_router.read_input(col, row, port) {
                        self.tiles[i].push_stream_input(port, stream_word.data);
                        words_routed += 1;
                    }
                }
            }
        }

        words_routed
    }

    /// Step the complete data movement system.
    ///
    /// This steps DMA, routes between DMA and streams, and steps the stream router.
    /// Use this for a complete simulation cycle.
    ///
    /// The execution order is:
    /// 1. Step all DMA engines (transfers data to/from tile memory)
    /// 2. Route DMA MM2S output → StreamRouter (for global tile-to-tile routing)
    /// 3. Route DMA MM2S output → TileSwitch DMA slaves (for intra-tile local routing)
    /// 4. Step global stream router (data moves between tiles)
    /// 5. Route StreamRouter → TileSwitch external slaves (data enters tiles)
    /// 6. Step per-tile stream switches (intra-tile input → output forwarding)
    /// 7. Route TileSwitch DMA masters → DMA S2MM (for local route → DMA scenarios)
    /// 8. Route TileSwitch external masters → StreamRouter (data exits tiles)
    /// 9. Route StreamRouter → DMA S2MM (for direct global routing)
    ///
    /// Returns (dma_active, streams_moved, words_routed)
    pub fn step_data_movement(&mut self, host_memory: &mut HostMemory) -> (bool, bool, usize) {
        // Phase 1: Step all DMA engines
        let dma_active = self.step_all_dma(host_memory);

        // Phase 2: Route DMA MM2S output → StreamRouter (for global tile-to-tile routing)
        let mut words_routed = self.route_dma_to_stream();

        // Phase 3: Route DMA MM2S output → TileSwitch DMA slaves (for intra-tile local routing)
        // This handles cases like MemTile DMA MM2S → local route → North master
        words_routed += self.route_dma_to_tile_switches();

        // Phase 4: Step global stream router (tile-to-tile routing)
        let streams_moved = self.step_streams();

        // Phase 5: Route StreamRouter input → tile stream_switch.slaves (external ports only)
        // Data from other tiles enters via North/South slave ports
        words_routed += self.route_router_to_tile_switches();

        // Phase 6: Step per-tile stream switches (intra-tile routing)
        let tile_forwards = self.step_tile_switches();
        words_routed += tile_forwards;

        // Phase 7: Route TileSwitch DMA masters → DMA S2MM (for local route → DMA scenarios)
        // This handles cases like MemTile: South slave → local route → DMA S2MM
        words_routed += self.route_tile_switches_to_dma();

        // Phase 8: Route tile stream_switch external masters → StreamRouter (for outgoing)
        words_routed += self.route_tile_switches_to_router();

        // Phase 9: Route StreamRouter → DMA S2MM input (for direct global routing)
        // This MUST happen after step_streams so data is available
        words_routed += self.route_stream_to_dma();

        (dma_active, streams_moved || tile_forwards > 0, words_routed)
    }

    /// Route data from global stream router to per-tile stream switches.
    ///
    /// For tiles with configured local routes (like MemTile), this moves data
    /// from the global router's input FIFOs to the tile's stream_switch.slaves.
    /// Only routes external (North/South) ports - DMA slave ports receive from DMA engine.
    fn route_router_to_tile_switches(&mut self) -> usize {
        use super::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            let col = (i / self.rows as usize) as u8;
            let row = (i % self.rows as usize) as u8;

            // Only process tiles with local routes configured (e.g., MemTile)
            if self.tiles[i].stream_switch.local_route_count() == 0 {
                continue;
            }

            // Check each slave port for incoming data from global router
            let num_slaves = self.tiles[i].stream_switch.slaves.len();
            for slave_port in 0..num_slaves {
                // Only route external ports (North/South) from global router
                // DMA slave ports receive from DMA MM2S, not from global router
                let port_type = &self.tiles[i].stream_switch.slaves[slave_port].port_type;
                let is_external = matches!(port_type, PortType::North | PortType::South);

                // Debug: check if router has data for this port
                if self.stream_router.input_has_data(col, row, slave_port as u8) {
                    log::debug!("route_router_to_tile_switches: ({},{}) port {} has data, is_external={}, type={:?}",
                        col, row, slave_port, is_external, port_type);
                }

                if !is_external {
                    continue;
                }

                // Read from global router input FIFO for this tile+port
                if let Some(stream_word) = self.stream_router.read_input(col, row, slave_port as u8) {
                    // Push to tile's stream switch slave port
                    if self.tiles[i].stream_switch.slaves[slave_port].push(stream_word.data) {
                        words_routed += 1;
                        log::info!("Router->TileSwitch: ({},{}) slave[{}] <- 0x{:08X}",
                            col, row, slave_port, stream_word.data);
                    }
                }
            }
        }

        words_routed
    }

    /// Route DMA MM2S output to per-tile stream switch DMA slave ports.
    ///
    /// For tiles with local routes, DMA MM2S output goes to DMA slave ports
    /// so local routes can forward to external master ports.
    fn route_dma_to_tile_switches(&mut self) -> usize {
        use super::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Only process tiles with local routes configured
            if self.tiles[i].stream_switch.local_route_count() == 0 {
                continue;
            }

            // Pop data from DMA stream_out and push to tile switch slave ports
            while let Some(data) = self.dma_engines[i].pop_stream_out() {
                let tile_type = self.tiles[i].tile_type;
                let col = self.tiles[i].col;
                let row = self.tiles[i].row;

                // Map DMA MM2S channel to tile switch slave port based on tile type
                let slave_port = match tile_type {
                    TileType::MemTile => {
                        // MemTile MM2S channels 6-11 map to DMA slave ports 0-5
                        (if data.channel >= 6 { data.channel - 6 } else { data.channel }) as usize
                    }
                    TileType::Shim => {
                        // Shim MM2S output goes to South slave ports (connected to NoC/DDR)
                        // Per AM025 PL_MODULE: Slave ports 2-9 are South 0-7
                        // Find the South slave port that's configured to route to a North master
                        // This is determined by the CDO local route configuration
                        let mut target_slave: Option<usize> = None;
                        for route in &self.tiles[i].stream_switch.local_routes {
                            let master_idx = route.master_idx as usize;
                            if master_idx < self.tiles[i].stream_switch.masters.len() {
                                let master_type = &self.tiles[i].stream_switch.masters[master_idx].port_type;
                                // If master is North type, use the configured slave
                                if matches!(master_type, PortType::North) {
                                    target_slave = Some(route.slave_idx as usize);
                                    break;
                                }
                            }
                        }
                        target_slave.unwrap_or(2) // Default to slave[2] (South0) if not found
                    }
                    _ => {
                        // Compute tile MM2S channels 2-3 map to DMA slave ports 1-2 (per COMPUTE_SLAVE_PORTS)
                        // DMA0 is slave[1], DMA1 is slave[2]
                        // ch2 -> slave[1], ch3 -> slave[2]
                        (if data.channel >= 2 { data.channel - 1 } else { data.channel + 1 }) as usize
                    }
                };

                if slave_port < self.tiles[i].stream_switch.slaves.len() {
                    // Clone port type to avoid borrow conflict
                    let port_type = self.tiles[i].stream_switch.slaves[slave_port].port_type.clone();

                    // For Shim tiles, accept South ports; for others, accept DMA ports
                    let accept = match tile_type {
                        TileType::Shim => matches!(port_type, PortType::South),
                        _ => matches!(port_type, PortType::Dma(_)),
                    };

                    if accept {
                        if self.tiles[i].stream_switch.slaves[slave_port].push(data.data) {
                            words_routed += 1;
                            log::info!("DMA->TileSwitch: tile ({},{}) slave[{}] <- 0x{:08X} (ch {}, type={:?})",
                                col, row, slave_port, data.data, data.channel, port_type);
                        }
                    } else {
                        log::debug!("DMA->TileSwitch: tile ({},{}) slave[{}] rejected - wrong type {:?}",
                            col, row, slave_port, port_type);
                    }
                }
            }
        }

        words_routed
    }

    /// Route per-tile stream switch DMA master output to DMA S2MM input.
    ///
    /// For tiles with local routes, DMA master port output goes to DMA S2MM stream_in.
    fn route_tile_switches_to_dma(&mut self) -> usize {
        use super::stream_switch::PortType;
        use super::dma::StreamData;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Only process tiles with local routes configured
            if self.tiles[i].stream_switch.local_route_count() == 0 {
                continue;
            }

            // Check DMA master ports for outgoing data
            let num_masters = self.tiles[i].stream_switch.masters.len();
            for master_port in 0..num_masters {
                // Only route DMA master ports to DMA engine
                // Clone channel number before mutable borrow
                let dma_channel = match &self.tiles[i].stream_switch.masters[master_port].port_type {
                    PortType::Dma(ch) => Some(*ch),
                    _ => None,
                };

                if let Some(ch) = dma_channel {
                    if let Some(data) = self.tiles[i].stream_switch.masters[master_port].pop() {
                        // Push to DMA S2MM stream_in
                        let stream_data = StreamData {
                            data,
                            tlast: false, // TODO: Track TLAST properly
                            channel: ch,
                        };
                        if self.dma_engines[i].push_stream_in(stream_data) {
                            words_routed += 1;
                            log::trace!("TileSwitch->DMA: tile ({},{}) master[{}] -> DMA ch {} = 0x{:08X}",
                                self.tiles[i].col, self.tiles[i].row, master_port, ch, data);
                        }
                    }
                }
            }
        }

        words_routed
    }

    /// Route data from per-tile stream switches to global stream router.
    ///
    /// For tiles with configured local routes (like MemTile), this moves data
    /// from the tile's stream_switch.masters to the global router's output FIFOs.
    /// Only routes external (North/South) ports - DMA ports go to DMA engine.
    fn route_tile_switches_to_router(&mut self) -> usize {
        use super::stream_router::StreamWord;
        use super::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            let col = (i / self.rows as usize) as u8;
            let row = (i % self.rows as usize) as u8;

            // Only process tiles with local routes configured (e.g., MemTile)
            if self.tiles[i].stream_switch.local_route_count() == 0 {
                continue;
            }

            // Check each master port for outgoing data
            let num_masters = self.tiles[i].stream_switch.masters.len();
            for master_port in 0..num_masters {
                // Only route external ports (North/South) to global router
                // DMA master ports go to DMA S2MM, not to global router
                let port_type = &self.tiles[i].stream_switch.masters[master_port].port_type;
                let is_external = matches!(port_type, PortType::North | PortType::South);

                if !is_external {
                    // For DMA master ports, data goes to DMA S2MM (handled elsewhere)
                    continue;
                }

                // Pop from tile's stream switch master port
                if let Some(data) = self.tiles[i].stream_switch.masters[master_port].pop() {
                    // Push to global router output FIFO for this tile+port
                    let stream_word = StreamWord { data, tlast: false };
                    if self.stream_router.write_output(col, row, master_port as u8, stream_word) {
                        words_routed += 1;
                        log::trace!("TileSwitch->Router: ({},{}) master[{}] -> 0x{:08X}",
                            col, row, master_port, data);
                    }
                }
            }
        }

        words_routed
    }

    /// Count tiles by type.
    pub fn count_by_type(&self) -> (usize, usize, usize) {
        let mut shim = 0;
        let mut mem = 0;
        let mut compute = 0;
        for tile in &self.tiles {
            match tile.tile_type {
                TileType::Shim => shim += 1,
                TileType::MemTile => mem += 1,
                TileType::Compute => compute += 1,
            }
        }
        (shim, mem, compute)
    }

    /// Reset all tiles to initial state.
    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.core.reset();
            for lock in &mut tile.locks {
                lock.value = 0;
            }
            for bd in &mut tile.dma_bds {
                *bd = Default::default();
            }
            for ch in &mut tile.dma_channels {
                *ch = Default::default();
            }
            // Recreate stream switch based on tile type (they have different port configurations)
            tile.stream_switch = match tile.tile_type {
                TileType::Shim => super::stream_switch::StreamSwitch::new_shim_tile(tile.col),
                TileType::MemTile => super::stream_switch::StreamSwitch::new_mem_tile(tile.col, tile.row),
                TileType::Compute => super::stream_switch::StreamSwitch::new_compute_tile(tile.col, tile.row),
            };
            // Note: We don't zero memory here for performance
            // Call zero_memory() explicitly if needed
        }

        // Reset all DMA engines
        for engine in &mut self.dma_engines {
            engine.reset();
        }

        // Reset stream router
        self.stream_router.reset();
    }

    /// Zero all tile memory (slow, use only during initialization).
    pub fn zero_memory(&mut self) {
        for tile in &mut self.tiles {
            tile.data_memory_mut().fill(0);
            if let Some(pm) = tile.program_memory_mut() {
                pm.fill(0);
            }
        }
    }

    /// Print array summary.
    pub fn print_summary(&self) {
        println!("Tile Array: {} ({} cols × {} rows)", self.arch, self.cols, self.rows);
        let (shim, mem, compute) = self.count_by_type();
        println!("  Shim tiles: {}", shim);
        println!("  Memory tiles: {}", mem);
        println!("  Compute tiles: {}", compute);
        println!("  Total: {} tiles", self.tiles.len());
    }
}

impl std::fmt::Debug for TileArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileArray")
            .field("arch", &self.arch)
            .field("cols", &self.cols)
            .field("rows", &self.rows)
            .field("tiles", &self.tiles.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation_npu1() {
        let array = TileArray::npu1();
        assert_eq!(array.cols(), 5);
        assert_eq!(array.rows(), 6);
        assert_eq!(array.tiles.len(), 30);

        let (shim, mem, compute) = array.count_by_type();
        assert_eq!(shim, 5);  // Row 0, all columns
        assert_eq!(mem, 5);   // Row 1, all columns
        assert_eq!(compute, 20); // Rows 2-5, all columns
    }

    #[test]
    fn test_tile_access() {
        let mut array = TileArray::npu1();

        // Access by coordinates
        let tile = array.get(1, 2).unwrap();
        assert_eq!(tile.col, 1);
        assert_eq!(tile.row, 2);
        assert!(tile.is_compute());

        // Modify tile
        let tile = array.get_mut(1, 2).unwrap();
        tile.core.pc = 0x100;
        assert_eq!(array.tile(1, 2).core.pc, 0x100);
    }

    #[test]
    fn test_out_of_bounds() {
        let array = TileArray::npu1();
        assert!(array.get(10, 0).is_none());
        assert!(array.get(0, 10).is_none());
    }

    #[test]
    fn test_tile_types() {
        let array = TileArray::npu1();

        // Shim tiles at row 0
        assert!(array.tile(0, 0).is_shim());
        assert!(array.tile(4, 0).is_shim());

        // Mem tiles at row 1
        assert!(array.tile(0, 1).is_mem_tile());
        assert!(array.tile(4, 1).is_mem_tile());

        // Compute tiles at rows 2-5
        assert!(array.tile(0, 2).is_compute());
        assert!(array.tile(4, 5).is_compute());
    }

    #[test]
    fn test_compute_tile_iteration() {
        let array = TileArray::npu1();
        let compute_count = array.compute_tiles().count();
        assert_eq!(compute_count, 20); // 5 cols × 4 rows (2-5)
    }

    #[test]
    fn test_reset() {
        let mut array = TileArray::npu1();

        // Modify some state
        array.tile_mut(1, 2).core.pc = 0x1000;
        array.tile_mut(1, 2).locks[0].value = 5;

        // Reset
        array.reset();

        // Verify reset
        assert_eq!(array.tile(1, 2).core.pc, 0);
        assert_eq!(array.tile(1, 2).locks[0].value, 0);
    }

    // === DMA Engine Integration Tests ===

    #[test]
    fn test_dma_engine_creation() {
        let array = TileArray::npu1();

        // Each tile should have a DMA engine
        assert_eq!(array.dma_engines.len(), 30);

        // Compute tile (row 2+) should have 4 channels
        let engine = array.dma_engine(1, 2).unwrap();
        assert_eq!(engine.num_channels(), 4);

        // Memory tile (row 1) should have 12 channels
        let engine = array.dma_engine(1, 1).unwrap();
        assert_eq!(engine.num_channels(), 12);
    }

    #[test]
    fn test_dma_engine_access() {
        let mut array = TileArray::npu1();

        // Get mutable engine and configure it
        let engine = array.dma_engine_mut(2, 3).unwrap();
        assert_eq!(engine.col, 2);
        assert_eq!(engine.row, 3);
    }

    #[test]
    fn test_tile_and_dma() {
        let mut array = TileArray::npu1();

        // Get both tile and DMA engine
        let (tile, engine) = array.tile_and_dma(3, 4).unwrap();
        assert_eq!(tile.col, 3);
        assert_eq!(tile.row, 4);
        assert_eq!(engine.col, 3);
        assert_eq!(engine.row, 4);
    }

    #[test]
    fn test_no_active_dma_initially() {
        let array = TileArray::npu1();
        assert!(!array.any_dma_active());
    }

    #[test]
    fn test_dma_reset() {
        use crate::device::dma::BdConfig;

        let mut array = TileArray::npu1();

        // Configure and start a DMA transfer
        let engine = array.dma_engine_mut(1, 2).unwrap();
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
        engine.start_channel(0, 0).unwrap();
        assert!(engine.channel_active(0));

        // Reset should clear it
        array.reset();
        let engine = array.dma_engine(1, 2).unwrap();
        assert!(!engine.any_channel_active());
    }
}
