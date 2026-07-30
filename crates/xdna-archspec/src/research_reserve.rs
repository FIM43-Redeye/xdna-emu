//! Versioned NPU1 research-reserve ledger and retirement report.
//!
//! The open-source architecture coverage model supplies one input. External
//! evidence remains untrusted until a later bundle validator audits it.

use crate::types::Architecture;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::{Component, Path},
};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReserveLedger {
    pub schema_version: u32,
    pub tuples: Vec<PinnedTuple>,
    pub inventory: Vec<InventoryEntry>,
    pub facts: Vec<HardwareFact>,
    pub evidence: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedTuple {
    pub id: String,
    pub title: String,
    #[serde(with = "architecture_serde")]
    pub architecture: Architecture,
    pub device: DevicePin,
    pub firmware: ContentPin,
    pub driver_surface: RevisionPin,
    pub kernel_corpus_evidence_ids: Vec<String>,
    pub identity_state: TupleIdentityState,
    pub inventory_scope: InventoryScope,
    pub live_attestation_evidence_ids: Vec<String>,
    pub offline_rehearsal_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePin {
    pub vendor_id: String,
    pub device_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentPin {
    pub logical_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionPin {
    pub repository: String,
    pub commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum TupleIdentityState {
    Open { missing_fields: Vec<String> },
    Complete { evidence_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum InventoryScope {
    Open { remaining_sources: Vec<String> },
    Sealed { evidence_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryEntry {
    pub id: String,
    pub title: String,
    pub source_refs: Vec<String>,
    pub tuple_ids: Vec<String>,
    pub coverage_domain_ids: Vec<String>,
    pub dependency_ids: Vec<String>,
    pub disposition: InventoryDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum InventoryDisposition {
    Applicable { fact_ids: Vec<String> },
    ProvenNotApplicable { fact_ids: Vec<String> },
    Deferred { reason: String },
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFact {
    pub id: String,
    pub statement: String,
    pub tuple_ids: Vec<String>,
    pub dependency_fact_ids: Vec<String>,
    pub preconditions: Vec<String>,
    pub initial_state: Vec<String>,
    pub stimulus: Vec<String>,
    pub external_events: Vec<String>,
    pub expected_transition: String,
    pub expected_outputs: Vec<String>,
    pub ordering: Vec<String>,
    pub timing_bounds: Vec<String>,
    pub supporting_evidence_ids: Vec<String>,
    pub control_evidence_ids: Vec<String>,
    pub counterevidence_ids: Vec<String>,
    pub alternatives_ruled_out: Vec<String>,
    pub remaining_unknowns: Vec<String>,
    pub source_refs: Vec<String>,
    pub implementation_refs: Vec<String>,
    pub test_refs: Vec<String>,
    pub promotion: PromotionState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PromotionState {
    Observed,
    Derived,
    Verified,
    Encoded,
    RetirementQualified,
    Contested {
        reason: String,
        evidence_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    pub kind: EvidenceKind,
    pub candidate_tuple_ids: Vec<String>,
    pub location: StableLocation,
    pub intake_refs: Vec<String>,
    pub expected_digests: EvidenceDigests,
    pub provenance_gaps: Vec<String>,
    pub retention: RetentionClass,
    pub redistributability: Redistributability,
    pub expected_replicas: Vec<ExpectedReplica>,
    pub preservation_notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    ToolchainSource,
    HardwareWitness,
    HistoricalEmulatorWitness,
    ImplementationFixture,
    Documentation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StableLocation {
    pub alias: String,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceDigests {
    pub metadata_fingerprint_sha256: Option<String>,
    pub checksum_index_sha256: Option<String>,
    pub manifest_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    WorkingCapture,
    WitnessCapture,
    ImplementationFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Redistributability {
    Redistributable,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedReplica {
    pub id: String,
    pub location: StableLocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug)]
pub struct LedgerError {
    pub issues: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseCheckKind {
    TupleIdentity,
    Inventory,
    Fact,
    Implementation,
    Evidence,
    Replica,
    SemanticProvenance,
    LiveAttestation,
    OfflineRehearsal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerCode {
    TupleIdentityOpen,
    InventoryScopeOpen,
    InventoryDeferred,
    InventoryUnknown,
    InventoryFactUnqualified,
    FactNotRetirementQualified,
    FactContested,
    FactContractIncomplete,
    FactUnknownsOpen,
    FactSupportingEvidenceMissing,
    FactControlEvidenceMissing,
    FactAlternativesMissing,
    ImplementationMissing,
    TestsMissing,
    EvidenceLegacyIncomplete,
    EvidenceProvenanceIncomplete,
    EvidenceUnaudited,
    ReplicaInsufficient,
    SemanticProvenanceOpen,
    LiveAttestationMissing,
    LiveAttestationUnaudited,
    OfflineRehearsalMissing,
    OfflineRehearsalUnaudited,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ReleaseBlocker {
    pub check: ReleaseCheckKind,
    pub code: BlockerCode,
    pub record_id: Option<String>,
    pub dependency_path: Vec<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReleaseCheck {
    pub kind: ReleaseCheckKind,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReleaseReport {
    pub tuple_id: String,
    pub checks: Vec<ReleaseCheck>,
    pub blockers: Vec<ReleaseBlocker>,
    pub is_clean: bool,
}

#[derive(Debug, Default)]
struct EvidenceAudit {
    verified_evidence_ids: BTreeSet<String>,
    verified_replica_ids: BTreeSet<(String, String)>,
}

#[derive(Debug)]
struct EvaluationInputs {
    semantic_provenance_clean: bool,
    evidence_audit: EvidenceAudit,
}

impl ReserveLedger {
    pub fn npu1() -> Result<Self, LedgerError> {
        Self::from_json(include_str!("../data/research-reserve/npu1.json"))
    }

    pub fn from_json(json: &str) -> Result<Self, LedgerError> {
        let ledger: Self = serde_json::from_str(json).map_err(|error| LedgerError {
            issues: vec![ValidationIssue { path: "$".into(), message: error.to_string() }],
        })?;
        let mut issues = ledger.validation_issues();
        if ledger.schema_version != SCHEMA_VERSION {
            issues.push(ValidationIssue {
                path: "$.schema_version".into(),
                message: format!("unsupported schema_version {}", ledger.schema_version),
            });
        }
        if issues.is_empty() {
            Ok(ledger)
        } else {
            issues.sort_by(|a, b| (&a.path, &a.message).cmp(&(&b.path, &b.message)));
            Err(LedgerError { issues })
        }
    }

    pub fn clean_release(&self, tuple_id: &str) -> Result<ReleaseReport, LedgerError> {
        let tuple = self.tuple(tuple_id)?;
        self.evaluate_release(
            tuple_id,
            &EvaluationInputs {
                semantic_provenance_clean: crate::coverage::CoverageModel::build(tuple.architecture)
                    .semantic_provenance_clean(),
                evidence_audit: EvidenceAudit::default(),
            },
        )
    }

    fn evaluate_release(
        &self,
        tuple_id: &str,
        inputs: &EvaluationInputs,
    ) -> Result<ReleaseReport, LedgerError> {
        let tuple = self.tuple(tuple_id)?;
        let facts: BTreeMap<&str, &HardwareFact> =
            self.facts.iter().map(|fact| (fact.id.as_str(), fact)).collect();
        let mut blockers = Vec::new();
        let mut evidence_ids = BTreeSet::new();

        evidence_ids.extend(tuple.kernel_corpus_evidence_ids.iter().map(String::as_str));
        evidence_ids.extend(tuple.live_attestation_evidence_ids.iter().map(String::as_str));
        evidence_ids.extend(tuple.offline_rehearsal_evidence_ids.iter().map(String::as_str));

        match &tuple.identity_state {
            TupleIdentityState::Open { missing_fields } => push_blocker(
                &mut blockers,
                ReleaseCheckKind::TupleIdentity,
                BlockerCode::TupleIdentityOpen,
                Some(&tuple.id),
                vec![tuple.id.clone()],
                format!("{} tuple identity fields remain open", missing_fields.len()),
            ),
            TupleIdentityState::Complete { evidence_ids: ids } => {
                evidence_ids.extend(ids.iter().map(String::as_str));
            }
        }

        match &tuple.inventory_scope {
            InventoryScope::Open { remaining_sources } => push_blocker(
                &mut blockers,
                ReleaseCheckKind::Inventory,
                BlockerCode::InventoryScopeOpen,
                Some(&tuple.id),
                vec![tuple.id.clone()],
                format!("{} inventory discovery sources remain open", remaining_sources.len()),
            ),
            InventoryScope::Sealed { evidence_ids: ids } => {
                evidence_ids.extend(ids.iter().map(String::as_str));
            }
        }

        for entry in self
            .inventory
            .iter()
            .filter(|entry| entry.tuple_ids.iter().any(|id| id == tuple_id))
        {
            match &entry.disposition {
                InventoryDisposition::Deferred { reason } => push_blocker(
                    &mut blockers,
                    ReleaseCheckKind::Inventory,
                    BlockerCode::InventoryDeferred,
                    Some(&entry.id),
                    vec![entry.id.clone()],
                    reason.clone(),
                ),
                InventoryDisposition::Unknown { reason } => push_blocker(
                    &mut blockers,
                    ReleaseCheckKind::Inventory,
                    BlockerCode::InventoryUnknown,
                    Some(&entry.id),
                    vec![entry.id.clone()],
                    reason.clone(),
                ),
                InventoryDisposition::Applicable { fact_ids }
                | InventoryDisposition::ProvenNotApplicable { fact_ids } => {
                    for fact_id in fact_ids {
                        let mut path = vec![entry.id.clone()];
                        if !audit_fact(&facts, fact_id, &mut path, &mut evidence_ids, &mut blockers) {
                            push_blocker(
                                &mut blockers,
                                ReleaseCheckKind::Inventory,
                                BlockerCode::InventoryFactUnqualified,
                                Some(&entry.id),
                                vec![entry.id.clone(), fact_id.clone()],
                                format!("fact `{fact_id}` does not satisfy the retirement contract"),
                            );
                        }
                    }
                }
            }
        }

        let evidence: BTreeMap<&str, &EvidenceRecord> =
            self.evidence.iter().map(|record| (record.id.as_str(), record)).collect();
        for evidence_id in evidence_ids {
            let record = evidence[evidence_id];
            if record.expected_digests.manifest_sha256.is_none() {
                push_blocker(
                    &mut blockers,
                    ReleaseCheckKind::Evidence,
                    BlockerCode::EvidenceLegacyIncomplete,
                    Some(&record.id),
                    vec![record.id.clone()],
                    "canonical manifest digest is unavailable".into(),
                );
            }
            if !record.provenance_gaps.is_empty() {
                push_blocker(
                    &mut blockers,
                    ReleaseCheckKind::Evidence,
                    BlockerCode::EvidenceProvenanceIncomplete,
                    Some(&record.id),
                    vec![record.id.clone()],
                    format!("{} provenance gaps remain", record.provenance_gaps.len()),
                );
            }
            if !inputs.evidence_audit.verified_evidence_ids.contains(&record.id) {
                push_blocker(
                    &mut blockers,
                    ReleaseCheckKind::Evidence,
                    BlockerCode::EvidenceUnaudited,
                    Some(&record.id),
                    vec![record.id.clone()],
                    "external evidence has not passed the trusted bundle audit".into(),
                );
            }
            if record.retention == RetentionClass::WitnessCapture {
                let verified_replicas = record
                    .expected_replicas
                    .iter()
                    .filter(|replica| {
                        inputs
                            .evidence_audit
                            .verified_replica_ids
                            .contains(&(record.id.clone(), replica.id.clone()))
                    })
                    .count();
                if verified_replicas < 2 {
                    push_blocker(
                        &mut blockers,
                        ReleaseCheckKind::Replica,
                        BlockerCode::ReplicaInsufficient,
                        Some(&record.id),
                        vec![record.id.clone()],
                        format!("verified independent replicas: {verified_replicas}/2"),
                    );
                }
            }
        }

        if !inputs.semantic_provenance_clean {
            push_blocker(
                &mut blockers,
                ReleaseCheckKind::SemanticProvenance,
                BlockerCode::SemanticProvenanceOpen,
                Some(&tuple.id),
                vec![tuple.id.clone()],
                "semantic perishable or comprehension queues remain open".into(),
            );
        }
        audit_evidence_use(
            &tuple.id,
            &tuple.live_attestation_evidence_ids,
            ReleaseCheckKind::LiveAttestation,
            BlockerCode::LiveAttestationMissing,
            BlockerCode::LiveAttestationUnaudited,
            &inputs.evidence_audit,
            &mut blockers,
        );
        audit_evidence_use(
            &tuple.id,
            &tuple.offline_rehearsal_evidence_ids,
            ReleaseCheckKind::OfflineRehearsal,
            BlockerCode::OfflineRehearsalMissing,
            BlockerCode::OfflineRehearsalUnaudited,
            &inputs.evidence_audit,
            &mut blockers,
        );

        blockers.sort();
        blockers.dedup();
        let checks = [
            ReleaseCheckKind::TupleIdentity,
            ReleaseCheckKind::Inventory,
            ReleaseCheckKind::Fact,
            ReleaseCheckKind::Implementation,
            ReleaseCheckKind::Evidence,
            ReleaseCheckKind::Replica,
            ReleaseCheckKind::SemanticProvenance,
            ReleaseCheckKind::LiveAttestation,
            ReleaseCheckKind::OfflineRehearsal,
        ]
        .into_iter()
        .map(|kind| ReleaseCheck { kind, passed: !blockers.iter().any(|blocker| blocker.check == kind) })
        .collect();
        let is_clean = blockers.is_empty();
        Ok(ReleaseReport { tuple_id: tuple.id.clone(), checks, blockers, is_clean })
    }

    fn tuple(&self, tuple_id: &str) -> Result<&PinnedTuple, LedgerError> {
        self.tuples
            .iter()
            .find(|tuple| tuple.id == tuple_id)
            .ok_or_else(|| LedgerError {
                issues: vec![ValidationIssue {
                    path: "$.tuples".into(),
                    message: format!("unknown tuple id `{tuple_id}`"),
                }],
            })
    }

    fn validation_issues(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let tuple_ids = validate_record_ids(
            "tuple",
            "tuples",
            "tuple",
            self.tuples.iter().map(|record| &record.id),
            &mut issues,
        );
        let inventory_ids = validate_record_ids(
            "inventory",
            "inventory",
            "inventory",
            self.inventory.iter().map(|record| &record.id),
            &mut issues,
        );
        let fact_ids = validate_record_ids(
            "fact",
            "facts",
            "fact",
            self.facts.iter().map(|record| &record.id),
            &mut issues,
        );
        let evidence_ids = validate_record_ids(
            "evidence",
            "evidence",
            "evidence",
            self.evidence.iter().map(|record| &record.id),
            &mut issues,
        );

        for (index, tuple) in self.tuples.iter().enumerate() {
            let base = format!("$.tuples[{index}]");
            require_text(&tuple.title, &format!("{base}.title"), "tuple title", &mut issues);
            require_text(
                &tuple.device.vendor_id,
                &format!("{base}.device.vendor_id"),
                "vendor id",
                &mut issues,
            );
            require_text(
                &tuple.device.device_id,
                &format!("{base}.device.device_id"),
                "device id",
                &mut issues,
            );
            require_text(
                &tuple.firmware.logical_name,
                &format!("{base}.firmware.logical_name"),
                "firmware logical name",
                &mut issues,
            );
            validate_sha256(&tuple.firmware.sha256, &format!("{base}.firmware.sha256"), &mut issues);
            require_text(
                &tuple.driver_surface.repository,
                &format!("{base}.driver_surface.repository"),
                "driver repository",
                &mut issues,
            );
            require_text(
                &tuple.driver_surface.commit,
                &format!("{base}.driver_surface.commit"),
                "driver commit",
                &mut issues,
            );
            validate_refs(
                &tuple.kernel_corpus_evidence_ids,
                &evidence_ids,
                &format!("{base}.kernel_corpus_evidence_ids"),
                "evidence",
                &mut issues,
            );
            match &tuple.identity_state {
                TupleIdentityState::Open { missing_fields } => {
                    require_nonempty(
                        missing_fields,
                        &format!("{base}.identity_state.missing_fields"),
                        "open tuple identity",
                        &mut issues,
                    );
                    validate_text_list(
                        missing_fields,
                        &format!("{base}.identity_state.missing_fields"),
                        &mut issues,
                    );
                }
                TupleIdentityState::Complete { evidence_ids: refs } => {
                    require_nonempty(
                        refs,
                        &format!("{base}.identity_state.evidence_ids"),
                        "complete tuple identity",
                        &mut issues,
                    );
                    validate_refs(
                        refs,
                        &evidence_ids,
                        &format!("{base}.identity_state.evidence_ids"),
                        "evidence",
                        &mut issues,
                    );
                }
            }
            match &tuple.inventory_scope {
                InventoryScope::Open { remaining_sources } => {
                    require_nonempty(
                        remaining_sources,
                        &format!("{base}.inventory_scope.remaining_sources"),
                        "open inventory scope",
                        &mut issues,
                    );
                    validate_text_list(
                        remaining_sources,
                        &format!("{base}.inventory_scope.remaining_sources"),
                        &mut issues,
                    );
                }
                InventoryScope::Sealed { evidence_ids: refs } => {
                    require_nonempty(
                        refs,
                        &format!("{base}.inventory_scope.evidence_ids"),
                        "sealed inventory scope",
                        &mut issues,
                    );
                    validate_refs(
                        refs,
                        &evidence_ids,
                        &format!("{base}.inventory_scope.evidence_ids"),
                        "evidence",
                        &mut issues,
                    );
                }
            }
            validate_refs(
                &tuple.live_attestation_evidence_ids,
                &evidence_ids,
                &format!("{base}.live_attestation_evidence_ids"),
                "evidence",
                &mut issues,
            );
            validate_refs(
                &tuple.offline_rehearsal_evidence_ids,
                &evidence_ids,
                &format!("{base}.offline_rehearsal_evidence_ids"),
                "evidence",
                &mut issues,
            );
        }

        for (index, entry) in self.inventory.iter().enumerate() {
            let base = format!("$.inventory[{index}]");
            require_text(&entry.title, &format!("{base}.title"), "inventory title", &mut issues);
            validate_text_list(&entry.source_refs, &format!("{base}.source_refs"), &mut issues);
            require_nonempty(
                &entry.tuple_ids,
                &format!("{base}.tuple_ids"),
                "inventory tuple scope",
                &mut issues,
            );
            validate_refs(&entry.tuple_ids, &tuple_ids, &format!("{base}.tuple_ids"), "tuple", &mut issues);
            for (domain_index, domain) in entry.coverage_domain_ids.iter().enumerate() {
                if !crate::coverage::spine_ids::SPINE_DOMAIN_IDS.contains(&domain.as_str()) {
                    issues.push(ValidationIssue {
                        path: format!("{base}.coverage_domain_ids[{domain_index}]"),
                        message: format!("unknown coverage domain `{domain}`"),
                    });
                }
            }
            validate_refs(
                &entry.dependency_ids,
                &inventory_ids,
                &format!("{base}.dependency_ids"),
                "inventory",
                &mut issues,
            );
            match &entry.disposition {
                InventoryDisposition::Applicable { fact_ids: refs }
                | InventoryDisposition::ProvenNotApplicable { fact_ids: refs } => {
                    if refs.is_empty() {
                        issues.push(ValidationIssue {
                            path: format!("{base}.disposition.fact_ids"),
                            message: "fact-bearing disposition requires at least one fact".into(),
                        });
                    }
                    validate_refs(
                        refs,
                        &fact_ids,
                        &format!("{base}.disposition.fact_ids"),
                        "fact",
                        &mut issues,
                    );
                }
                InventoryDisposition::Deferred { reason } | InventoryDisposition::Unknown { reason } => {
                    require_text(
                        reason,
                        &format!("{base}.disposition.reason"),
                        "disposition reason",
                        &mut issues,
                    );
                }
            }
        }

        for (index, fact) in self.facts.iter().enumerate() {
            let base = format!("$.facts[{index}]");
            require_text(&fact.statement, &format!("{base}.statement"), "fact statement", &mut issues);
            require_nonempty(&fact.tuple_ids, &format!("{base}.tuple_ids"), "fact tuple scope", &mut issues);
            validate_refs(&fact.tuple_ids, &tuple_ids, &format!("{base}.tuple_ids"), "tuple", &mut issues);
            validate_refs(
                &fact.dependency_fact_ids,
                &fact_ids,
                &format!("{base}.dependency_fact_ids"),
                "fact",
                &mut issues,
            );
            for (name, values) in [
                ("preconditions", &fact.preconditions),
                ("initial_state", &fact.initial_state),
                ("stimulus", &fact.stimulus),
                ("external_events", &fact.external_events),
                ("expected_outputs", &fact.expected_outputs),
                ("ordering", &fact.ordering),
                ("timing_bounds", &fact.timing_bounds),
                ("alternatives_ruled_out", &fact.alternatives_ruled_out),
                ("remaining_unknowns", &fact.remaining_unknowns),
                ("source_refs", &fact.source_refs),
                ("implementation_refs", &fact.implementation_refs),
                ("test_refs", &fact.test_refs),
            ] {
                validate_text_list(values, &format!("{base}.{name}"), &mut issues);
            }
            require_text(
                &fact.expected_transition,
                &format!("{base}.expected_transition"),
                "expected transition",
                &mut issues,
            );
            for (name, refs) in [
                ("supporting_evidence_ids", &fact.supporting_evidence_ids),
                ("control_evidence_ids", &fact.control_evidence_ids),
                ("counterevidence_ids", &fact.counterevidence_ids),
            ] {
                validate_refs(refs, &evidence_ids, &format!("{base}.{name}"), "evidence", &mut issues);
            }
            if let PromotionState::Contested { reason, evidence_ids: refs } = &fact.promotion {
                require_text(reason, &format!("{base}.promotion.reason"), "contested reason", &mut issues);
                require_nonempty(
                    refs,
                    &format!("{base}.promotion.evidence_ids"),
                    "contested fact",
                    &mut issues,
                );
                validate_refs(
                    refs,
                    &evidence_ids,
                    &format!("{base}.promotion.evidence_ids"),
                    "evidence",
                    &mut issues,
                );
            }
        }

        for (index, evidence) in self.evidence.iter().enumerate() {
            let base = format!("$.evidence[{index}]");
            require_nonempty(
                &evidence.candidate_tuple_ids,
                &format!("{base}.candidate_tuple_ids"),
                "evidence tuple scope",
                &mut issues,
            );
            validate_refs(
                &evidence.candidate_tuple_ids,
                &tuple_ids,
                &format!("{base}.candidate_tuple_ids"),
                "tuple",
                &mut issues,
            );
            validate_location(&evidence.location, &format!("{base}.location"), &mut issues);
            validate_text_list(&evidence.intake_refs, &format!("{base}.intake_refs"), &mut issues);
            for (name, digest) in [
                (
                    "metadata_fingerprint_sha256",
                    evidence.expected_digests.metadata_fingerprint_sha256.as_deref(),
                ),
                ("checksum_index_sha256", evidence.expected_digests.checksum_index_sha256.as_deref()),
                ("manifest_sha256", evidence.expected_digests.manifest_sha256.as_deref()),
            ] {
                if let Some(digest) = digest {
                    validate_sha256(digest, &format!("{base}.expected_digests.{name}"), &mut issues);
                }
            }
            validate_text_list(&evidence.provenance_gaps, &format!("{base}.provenance_gaps"), &mut issues);
            validate_text_list(
                &evidence.preservation_notes,
                &format!("{base}.preservation_notes"),
                &mut issues,
            );
            let mut replica_ids = BTreeSet::new();
            for (replica_index, replica) in evidence.expected_replicas.iter().enumerate() {
                let replica_base = format!("{base}.expected_replicas[{replica_index}]");
                if !valid_dotted_id(&replica.id) {
                    issues.push(ValidationIssue {
                        path: format!("{replica_base}.id"),
                        message: format!("invalid replica id `{}`", replica.id),
                    });
                } else if !replica_ids.insert(replica.id.as_str()) {
                    issues.push(ValidationIssue {
                        path: format!("{replica_base}.id"),
                        message: format!("duplicate replica id `{}`", replica.id),
                    });
                }
                validate_location(&replica.location, &format!("{replica_base}.location"), &mut issues);
            }
        }

        let inventory_graph = self
            .inventory
            .iter()
            .map(|entry| (entry.id.clone(), entry.dependency_ids.clone()))
            .collect();
        if let Some(cycle) = dependency_cycle(&inventory_graph) {
            issues.push(ValidationIssue {
                path: "$.inventory".into(),
                message: format!("inventory dependency cycle: {}", cycle.join(" -> ")),
            });
        }
        let fact_graph = self
            .facts
            .iter()
            .map(|fact| (fact.id.clone(), fact.dependency_fact_ids.clone()))
            .collect();
        if let Some(cycle) = dependency_cycle(&fact_graph) {
            issues.push(ValidationIssue {
                path: "$.facts".into(),
                message: format!("fact dependency cycle: {}", cycle.join(" -> ")),
            });
        }

        issues
    }
}

pub fn render_release_report(ledger: &ReserveLedger, report: &ReleaseReport) -> String {
    let tuple = ledger
        .tuples
        .iter()
        .find(|tuple| tuple.id == report.tuple_id)
        .expect("release report must reference a validated ledger tuple");
    let mut inventory = ledger
        .inventory
        .iter()
        .filter(|entry| entry.tuple_ids.contains(&tuple.id))
        .collect::<Vec<_>>();
    let mut facts = ledger
        .facts
        .iter()
        .filter(|fact| fact.tuple_ids.contains(&tuple.id))
        .collect::<Vec<_>>();
    let mut evidence = ledger
        .evidence
        .iter()
        .filter(|record| record.candidate_tuple_ids.contains(&tuple.id))
        .collect::<Vec<_>>();
    inventory.sort_by_key(|entry| &entry.id);
    facts.sort_by_key(|fact| &fact.id);
    evidence.sort_by_key(|record| &record.id);

    let mut lines = vec![
        "# NPU1 release report".into(),
        String::new(),
        "Generated by `cargo run -p xdna-archspec --example gen_coverage_artifacts`. Do not hand-edit."
            .into(),
        String::new(),
        format!("**Result: {}**", if report.is_clean { "CLEAN" } else { "BLOCKED" }),
        String::new(),
        "## Pinned tuple".into(),
        String::new(),
        format!("- ID: `{}`", tuple.id),
        format!("- Architecture: `{}`", tuple.architecture),
        format!("- Device: `{}:{}`", tuple.device.vendor_id, tuple.device.device_id),
        format!("- Firmware: `{}`", tuple.firmware.logical_name),
        format!("- Firmware SHA-256: `{}`", tuple.firmware.sha256),
        format!(
            "- Driver surface: `{}` at `{}`",
            tuple.driver_surface.repository, tuple.driver_surface.commit
        ),
    ];
    match &tuple.identity_state {
        TupleIdentityState::Open { missing_fields } => {
            lines.push("- Identity: **OPEN**".into());
            lines.extend(missing_fields.iter().map(|field| format!("  - Missing: {field}")));
        }
        TupleIdentityState::Complete { evidence_ids } => {
            lines.push(format!("- Identity: **COMPLETE** ({})", inline_ids(evidence_ids)));
        }
    }
    match &tuple.inventory_scope {
        InventoryScope::Open { remaining_sources } => {
            lines.push("- Inventory scope: **OPEN**".into());
            lines.extend(remaining_sources.iter().map(|source| format!("  - Remaining source: {source}")));
        }
        InventoryScope::Sealed { evidence_ids } => {
            lines.push(format!("- Inventory scope: **SEALED** ({})", inline_ids(evidence_ids)));
        }
    }
    lines.push(format!("- Kernel corpus evidence: {}", inline_ids(&tuple.kernel_corpus_evidence_ids)));
    lines.push(format!("- Live attestation evidence: {}", inline_ids(&tuple.live_attestation_evidence_ids)));
    lines
        .push(format!("- Offline rehearsal evidence: {}", inline_ids(&tuple.offline_rehearsal_evidence_ids)));

    lines.extend([String::new(), "## Inventory".into(), String::new()]);
    if inventory.is_empty() {
        lines.push("_none_".into());
    }
    for entry in inventory {
        lines.push(format!("### `{}`", entry.id));
        lines.push(String::new());
        lines.push(entry.title.clone());
        lines.push(String::new());
        lines.push(format!("- Coverage domains: {}", inline_ids(&entry.coverage_domain_ids)));
        lines.push(format!("- Dependencies: {}", inline_ids(&entry.dependency_ids)));
        match &entry.disposition {
            InventoryDisposition::Applicable { fact_ids } => {
                lines.push(format!("- Disposition: `applicable` ({})", inline_ids(fact_ids)));
            }
            InventoryDisposition::ProvenNotApplicable { fact_ids } => {
                lines.push(format!("- Disposition: `proven_not_applicable` ({})", inline_ids(fact_ids)));
            }
            InventoryDisposition::Deferred { reason } => {
                lines.push(format!("- Disposition: `deferred` -- {reason}"));
            }
            InventoryDisposition::Unknown { reason } => {
                lines.push(format!("- Disposition: `unknown` -- {reason}"));
            }
        }
        lines.push(format!("- Sources: {}", inline_ids(&entry.source_refs)));
        lines.push(String::new());
    }

    lines.extend(["## Hardware facts".into(), String::new()]);
    if facts.is_empty() {
        lines.push("_none_".into());
    }
    for fact in facts {
        lines.push(format!("### `{}`", fact.id));
        lines.push(String::new());
        lines.push(fact.statement.clone());
        lines.push(String::new());
        lines.push(format!("- Promotion: `{}`", promotion_name(&fact.promotion)));
        lines.push(format!("- Dependencies: {}", inline_ids(&fact.dependency_fact_ids)));
        lines.push(format!("- Expected transition: {}", fact.expected_transition));
        lines.push(format!("- Supporting evidence: {}", inline_ids(&fact.supporting_evidence_ids)));
        lines.push(format!("- Control evidence: {}", inline_ids(&fact.control_evidence_ids)));
        lines.push(format!("- Implementation: {}", inline_ids(&fact.implementation_refs)));
        lines.push(format!("- Tests: {}", inline_ids(&fact.test_refs)));
        lines.push("- Remaining unknowns:".into());
        if fact.remaining_unknowns.is_empty() {
            lines.push("  - _none_".into());
        } else {
            lines.extend(fact.remaining_unknowns.iter().map(|unknown| format!("  - {unknown}")));
        }
        lines.push(String::new());
    }

    lines.extend(["## Evidence".into(), String::new()]);
    if evidence
        .iter()
        .any(|record| record.kind == EvidenceKind::HistoricalEmulatorWitness)
    {
        lines.push(
            "> **Non-promotion warning:** Historical emulator witnesses are regression evidence only; \
             they are not physical NPU evidence and cannot promote a fact without independent \
             corroboration."
                .into(),
        );
        lines.push(String::new());
    }
    if evidence.is_empty() {
        lines.push("_none_".into());
    }
    for record in evidence {
        lines.push(format!("### `{}`", record.id));
        lines.push(String::new());
        lines.push(format!("- Kind: `{}`", serialized_name(&record.kind)));
        lines.push(format!("- Location: `{}/{}`", record.location.alias, record.location.relative_path));
        lines.push(format!("- Intake references: {}", inline_ids(&record.intake_refs)));
        lines.push(format!("- Retention: `{}`", serialized_name(&record.retention)));
        lines.push(format!("- Redistributability: `{}`", serialized_name(&record.redistributability)));
        lines.push(format!(
            "- Metadata fingerprint SHA-256: {}",
            optional_digest(record.expected_digests.metadata_fingerprint_sha256.as_deref())
        ));
        lines.push(format!(
            "- Checksum index SHA-256: {}",
            optional_digest(record.expected_digests.checksum_index_sha256.as_deref())
        ));
        lines.push(format!(
            "- Manifest SHA-256: {}",
            optional_digest(record.expected_digests.manifest_sha256.as_deref())
        ));
        lines.push(format!("- Expected independent replicas: {}", record.expected_replicas.len()));
        lines.extend(record.expected_replicas.iter().map(|replica| {
            format!("  - `{}` at `{}/{}`", replica.id, replica.location.alias, replica.location.relative_path)
        }));
        lines.push("- Provenance gaps:".into());
        if record.provenance_gaps.is_empty() {
            lines.push("  - _none_".into());
        } else {
            lines.extend(record.provenance_gaps.iter().map(|gap| format!("  - {gap}")));
        }
        lines.push("- Preservation notes:".into());
        if record.preservation_notes.is_empty() {
            lines.push("  - _none_".into());
        } else {
            lines.extend(record.preservation_notes.iter().map(|note| format!("  - {note}")));
        }
        lines.push(String::new());
    }

    lines.extend([
        "## Release checks".into(),
        String::new(),
        "| Check | Result |".into(),
        "|-------|--------|".into(),
    ]);
    lines.extend(report.checks.iter().map(|check| {
        format!("| `{}` | {} |", serialized_name(&check.kind), if check.passed { "PASS" } else { "BLOCKED" })
    }));

    lines.extend([String::new(), "## Blockers".into(), String::new()]);
    if report.blockers.is_empty() {
        lines.push("_none_".into());
    } else {
        for blocker in &report.blockers {
            lines.push(format!(
                "- `{}` (`{}`), record {}, path: {} -- {}",
                serialized_name(&blocker.code),
                serialized_name(&blocker.check),
                blocker
                    .record_id
                    .as_deref()
                    .map(|id| format!("`{id}`"))
                    .unwrap_or_else(|| "_none_".into()),
                if blocker.dependency_path.is_empty() {
                    "_none_".into()
                } else {
                    blocker.dependency_path.join(" -> ")
                },
                blocker.detail
            ));
        }
    }
    format!("{}\n", lines.join("\n"))
}

fn inline_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        "_none_".into()
    } else {
        ids.iter().map(|id| format!("`{id}`")).collect::<Vec<_>>().join(", ")
    }
}

fn optional_digest(digest: Option<&str>) -> String {
    digest.map(|digest| format!("`{digest}`")).unwrap_or_else(|| "_missing_".into())
}

fn promotion_name(promotion: &PromotionState) -> &'static str {
    match promotion {
        PromotionState::Observed => "observed",
        PromotionState::Derived => "derived",
        PromotionState::Verified => "verified",
        PromotionState::Encoded => "encoded",
        PromotionState::RetirementQualified => "retirement_qualified",
        PromotionState::Contested { .. } => "contested",
    }
}

fn serialized_name(value: &impl Serialize) -> String {
    serde_json::to_value(value)
        .expect("release report enum must serialize")
        .as_str()
        .expect("release report enum must serialize as a string")
        .into()
}

fn audit_fact<'a>(
    facts: &BTreeMap<&'a str, &'a HardwareFact>,
    fact_id: &str,
    path: &mut Vec<String>,
    evidence_ids: &mut BTreeSet<&'a str>,
    blockers: &mut Vec<ReleaseBlocker>,
) -> bool {
    let fact = facts[fact_id];
    path.push(fact.id.clone());
    evidence_ids.extend(
        fact.supporting_evidence_ids
            .iter()
            .chain(&fact.control_evidence_ids)
            .chain(&fact.counterevidence_ids)
            .map(String::as_str),
    );
    if let PromotionState::Contested { evidence_ids: ids, .. } = &fact.promotion {
        evidence_ids.extend(ids.iter().map(String::as_str));
    }

    let mut qualified = matches!(fact.promotion, PromotionState::RetirementQualified);
    if !qualified {
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactNotRetirementQualified,
            Some(&fact.id),
            path.clone(),
            "declared promotion is not retirement_qualified".into(),
        );
    }
    if let PromotionState::Contested { reason, .. } = &fact.promotion {
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactContested,
            Some(&fact.id),
            path.clone(),
            reason.clone(),
        );
    }

    let missing_contract_fields = [
        ("preconditions", &fact.preconditions),
        ("initial_state", &fact.initial_state),
        ("stimulus", &fact.stimulus),
        ("external_events", &fact.external_events),
        ("expected_outputs", &fact.expected_outputs),
        ("ordering", &fact.ordering),
        ("timing_bounds", &fact.timing_bounds),
        ("source_refs", &fact.source_refs),
    ]
    .into_iter()
    .filter_map(|(name, values)| values.is_empty().then_some(name))
    .collect::<Vec<_>>();
    if !missing_contract_fields.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactContractIncomplete,
            Some(&fact.id),
            path.clone(),
            format!("missing {}", missing_contract_fields.join(", ")),
        );
    }
    if !fact.remaining_unknowns.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactUnknownsOpen,
            Some(&fact.id),
            path.clone(),
            format!("{} remaining unknowns", fact.remaining_unknowns.len()),
        );
    }
    if fact.supporting_evidence_ids.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactSupportingEvidenceMissing,
            Some(&fact.id),
            path.clone(),
            "supporting witness evidence is missing".into(),
        );
    }
    if fact.control_evidence_ids.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactControlEvidenceMissing,
            Some(&fact.id),
            path.clone(),
            "control evidence is missing".into(),
        );
    }
    if fact.alternatives_ruled_out.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Fact,
            BlockerCode::FactAlternativesMissing,
            Some(&fact.id),
            path.clone(),
            "alternatives have not been ruled out".into(),
        );
    }
    if fact.implementation_refs.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Implementation,
            BlockerCode::ImplementationMissing,
            Some(&fact.id),
            path.clone(),
            "emulator implementation reference is missing".into(),
        );
    }
    if fact.test_refs.is_empty() {
        qualified = false;
        push_blocker(
            blockers,
            ReleaseCheckKind::Implementation,
            BlockerCode::TestsMissing,
            Some(&fact.id),
            path.clone(),
            "executable test reference is missing".into(),
        );
    }

    for dependency_id in &fact.dependency_fact_ids {
        qualified &= audit_fact(facts, dependency_id, path, evidence_ids, blockers);
    }
    path.pop();
    qualified
}

fn audit_evidence_use(
    tuple_id: &str,
    evidence_ids: &[String],
    check: ReleaseCheckKind,
    missing_code: BlockerCode,
    unaudited_code: BlockerCode,
    audit: &EvidenceAudit,
    blockers: &mut Vec<ReleaseBlocker>,
) {
    if evidence_ids.is_empty() {
        push_blocker(
            blockers,
            check,
            missing_code,
            Some(tuple_id),
            vec![tuple_id.into()],
            "required evidence is missing".into(),
        );
        return;
    }
    for evidence_id in evidence_ids {
        if !audit.verified_evidence_ids.contains(evidence_id) {
            push_blocker(
                blockers,
                check,
                unaudited_code,
                Some(evidence_id),
                vec![tuple_id.into(), evidence_id.clone()],
                "referenced evidence has not passed the trusted bundle audit".into(),
            );
        }
    }
}

fn push_blocker(
    blockers: &mut Vec<ReleaseBlocker>,
    check: ReleaseCheckKind,
    code: BlockerCode,
    record_id: Option<&str>,
    dependency_path: Vec<String>,
    detail: String,
) {
    blockers.push(ReleaseBlocker {
        check,
        code,
        record_id: record_id.map(str::to_owned),
        dependency_path,
        detail,
    });
}

fn validate_record_ids<'a>(
    kind: &str,
    collection: &str,
    prefix: &str,
    ids: impl Iterator<Item = &'a String>,
    issues: &mut Vec<ValidationIssue>,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        let path = format!("$.{collection}[{index}].id");
        if !id.starts_with(&format!("{prefix}.")) || !valid_dotted_id(id) {
            issues.push(ValidationIssue { path, message: format!("invalid {kind} id `{id}`") });
        } else if !seen.insert(id.clone()) {
            issues.push(ValidationIssue { path, message: format!("duplicate {kind} id `{id}`") });
        }
    }
    seen
}

fn valid_dotted_id(id: &str) -> bool {
    !id.is_empty()
        && id.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            matches!(bytes.next(), Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit())
                && bytes.all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
                })
        })
}

fn require_text(value: &str, path: &str, label: &str, issues: &mut Vec<ValidationIssue>) {
    if value.trim().is_empty() {
        issues.push(ValidationIssue { path: path.into(), message: format!("{label} must not be blank") });
    }
}

fn require_nonempty<T>(values: &[T], path: &str, label: &str, issues: &mut Vec<ValidationIssue>) {
    if values.is_empty() {
        issues.push(ValidationIssue {
            path: path.into(),
            message: format!("{label} requires at least one value"),
        });
    }
}

fn validate_text_list(values: &[String], path: &str, issues: &mut Vec<ValidationIssue>) {
    for (index, value) in values.iter().enumerate() {
        require_text(value, &format!("{path}[{index}]"), "list entry", issues);
    }
}

fn validate_refs(
    refs: &[String],
    known: &BTreeSet<String>,
    path: &str,
    kind: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for (index, reference) in refs.iter().enumerate() {
        if !known.contains(reference) {
            issues.push(ValidationIssue {
                path: format!("{path}[{index}]"),
                message: format!("unknown {kind} id `{reference}`"),
            });
        }
    }
}

fn validate_sha256(value: &str, path: &str, issues: &mut Vec<ValidationIssue>) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')) {
        issues.push(ValidationIssue {
            path: path.into(),
            message: format!("invalid lowercase SHA-256 `{value}`"),
        });
    }
}

fn validate_location(location: &StableLocation, path: &str, issues: &mut Vec<ValidationIssue>) {
    require_text(&location.alias, &format!("{path}.alias"), "location alias", issues);
    let relative = Path::new(&location.relative_path);
    if location.relative_path.is_empty()
        || relative.is_absolute()
        || !relative.components().all(|component| matches!(component, Component::Normal(_)))
    {
        issues.push(ValidationIssue {
            path: format!("{path}.relative_path"),
            message: format!("invalid relative path `{}`", location.relative_path),
        });
    }
}

fn dependency_cycle(graph: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    fn visit(
        node: &str,
        graph: &BTreeMap<String, Vec<String>>,
        active: &mut Vec<String>,
        complete: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if let Some(start) = active.iter().position(|item| item == node) {
            let mut cycle = active[start..].to_vec();
            cycle.push(node.to_string());
            return Some(cycle);
        }
        if complete.contains(node) {
            return None;
        }
        active.push(node.to_string());
        if let Some(dependencies) = graph.get(node) {
            for dependency in dependencies {
                if graph.contains_key(dependency) {
                    if let Some(cycle) = visit(dependency, graph, active, complete) {
                        return Some(cycle);
                    }
                }
            }
        }
        active.pop();
        complete.insert(node.to_string());
        None
    }

    let mut complete = BTreeSet::new();
    for node in graph.keys() {
        if let Some(cycle) = visit(node, graph, &mut Vec::new(), &mut complete) {
            return Some(cycle);
        }
    }
    None
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, issue) in self.issues.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{}: {}", issue.path, issue.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for LedgerError {}

mod architecture_serde {
    use crate::types::Architecture;
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(architecture: &Architecture, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(match architecture {
            Architecture::Aie => "aie",
            Architecture::Aie2 => "aie2",
            Architecture::Aie2p => "aie2p",
        })
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Architecture, D::Error>
    where
        D: Deserializer<'de>,
    {
        match String::deserialize(deserializer)?.as_str() {
            "aie" => Ok(Architecture::Aie),
            "aie2" => Ok(Architecture::Aie2),
            "aie2p" => Ok(Architecture::Aie2p),
            other => Err(D::Error::custom(format!("unknown architecture `{other}`"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_LEDGER: &str = r#"
    {
      "schema_version": 1,
      "tuples": [{
        "id": "tuple.test.aie2",
        "title": "Test AIE2 tuple",
        "architecture": "aie2",
        "device": {
          "vendor_id": "1022",
          "device_id": "1502"
        },
        "firmware": {
          "logical_name": "firmware.bin",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        "driver_surface": {
          "repository": "amdxdna",
          "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "kernel_corpus_evidence_ids": [],
        "identity_state": {
          "state": "open",
          "missing_fields": ["host_kernel"]
        },
        "inventory_scope": {
          "state": "open",
          "remaining_sources": ["driver"]
        },
        "live_attestation_evidence_ids": [],
        "offline_rehearsal_evidence_ids": []
      }],
      "inventory": [],
      "facts": [],
      "evidence": []
    }"#;

    const LINKED_LEDGER: &str = r#"
    {
      "schema_version": 1,
      "tuples": [{
        "id": "tuple.test.aie2",
        "title": "Test AIE2 tuple",
        "architecture": "aie2",
        "device": {
          "vendor_id": "1022",
          "device_id": "1502"
        },
        "firmware": {
          "logical_name": "firmware.bin",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        "driver_surface": {
          "repository": "amdxdna",
          "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "kernel_corpus_evidence_ids": ["evidence.test.hw"],
        "identity_state": {
          "state": "complete",
          "evidence_ids": ["evidence.test.hw"]
        },
        "inventory_scope": {
          "state": "sealed",
          "evidence_ids": ["evidence.test.hw"]
        },
        "live_attestation_evidence_ids": ["evidence.test.hw"],
        "offline_rehearsal_evidence_ids": ["evidence.test.hw"]
      }],
      "inventory": [{
        "id": "inventory.test.command",
        "title": "Command execution",
        "source_refs": ["driver:aie2_msg.c"],
        "tuple_ids": ["tuple.test.aie2"],
        "coverage_domain_ids": ["dma"],
        "dependency_ids": [],
        "disposition": {
          "state": "applicable",
          "fact_ids": ["fact.test.command"]
        }
      }],
      "facts": [{
        "id": "fact.test.command",
        "statement": "A controlled command reaches completion.",
        "tuple_ids": ["tuple.test.aie2"],
        "dependency_fact_ids": [],
        "preconditions": ["context configured"],
        "initial_state": ["command idle"],
        "stimulus": ["submit command"],
        "external_events": ["none"],
        "expected_transition": "idle to complete",
        "expected_outputs": ["success"],
        "ordering": ["completion after submission"],
        "timing_bounds": ["finite"],
        "supporting_evidence_ids": ["evidence.test.hw"],
        "control_evidence_ids": ["evidence.test.hw"],
        "counterevidence_ids": [],
        "alternatives_ruled_out": ["host-only completion"],
        "remaining_unknowns": [],
        "source_refs": ["driver:aie2_msg.c"],
        "implementation_refs": ["src/firmware/mod.rs"],
        "test_refs": ["research_reserve::tests::release_closed"],
        "promotion": {
          "state": "retirement_qualified"
        }
      }],
      "evidence": [{
        "id": "evidence.test.hw",
        "kind": "hardware_witness",
        "candidate_tuple_ids": ["tuple.test.aie2"],
        "location": {
          "alias": "reserve",
          "relative_path": "bundles/test-hw"
        },
        "intake_refs": ["docs/evidence/test-hw.md"],
        "expected_digests": {
          "metadata_fingerprint_sha256": null,
          "checksum_index_sha256": null,
          "manifest_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        },
        "provenance_gaps": [],
        "retention": "witness_capture",
        "redistributability": "restricted",
        "expected_replicas": [{
          "id": "replica.test.one",
          "location": {
            "alias": "replica-one",
            "relative_path": "npu1/test-hw"
          }
        }, {
          "id": "replica.test.two",
          "location": {
            "alias": "replica-two",
            "relative_path": "npu1/test-hw"
          }
        }],
        "preservation_notes": []
      }]
    }"#;

    fn linked_value() -> serde_json::Value {
        serde_json::from_str(LINKED_LEDGER).expect("linked test ledger must be valid JSON")
    }

    fn replace(value: &mut serde_json::Value, pointer: &str, replacement: serde_json::Value) {
        *value
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("missing test pointer {pointer}")) = replacement;
    }

    fn assert_validation_error(value: serde_json::Value, expected: &str) {
        let json = serde_json::to_string(&value).expect("test ledger must serialize");
        let err = ReserveLedger::from_json(&json).expect_err("mutated ledger must fail validation");
        assert!(err.to_string().contains(expected), "expected `{expected}`, got `{err}`");
    }

    fn linked_ledger() -> ReserveLedger {
        ReserveLedger::from_json(LINKED_LEDGER).expect("linked ledger must validate")
    }

    fn verified_inputs() -> EvaluationInputs {
        EvaluationInputs {
            semantic_provenance_clean: true,
            evidence_audit: EvidenceAudit {
                verified_evidence_ids: BTreeSet::from(["evidence.test.hw".into()]),
                verified_replica_ids: BTreeSet::from([
                    ("evidence.test.hw".into(), "replica.test.one".into()),
                    ("evidence.test.hw".into(), "replica.test.two".into()),
                ]),
            },
        }
    }

    fn report_for(value: serde_json::Value, inputs: &EvaluationInputs) -> ReleaseReport {
        let json = serde_json::to_string(&value).unwrap();
        ReserveLedger::from_json(&json)
            .expect("mutated release fixture must remain structurally valid")
            .evaluate_release("tuple.test.aie2", inputs)
            .expect("known tuple must evaluate")
    }

    fn has_blocker(report: &ReleaseReport, code: BlockerCode) -> bool {
        report.blockers.iter().any(|blocker| blocker.code == code)
    }

    #[test]
    fn parse_rejects_unsupported_schema_version() {
        let input = MINIMAL_LEDGER.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
        let err = ReserveLedger::from_json(&input).expect_err("version 2 must fail closed");
        assert!(err.to_string().contains("unsupported schema_version 2"), "unexpected error: {err}");
    }

    #[test]
    fn parse_rejects_unknown_root_fields() {
        let input =
            MINIMAL_LEDGER.replacen("\"schema_version\": 1,", "\"schema_version\": 1, \"extra\": true,", 1);
        let err = ReserveLedger::from_json(&input).expect_err("unknown root field must fail closed");
        assert!(err.to_string().contains("unknown field `extra`"), "unexpected error: {err}");
    }

    #[test]
    fn parse_round_trips_architecture_as_snake_case() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).expect("minimal ledger must parse");
        assert_eq!(ledger.tuples[0].architecture, crate::types::Architecture::Aie2);

        let json = serde_json::to_value(&ledger).expect("ledger must serialize");
        assert_eq!(json["tuples"][0]["architecture"], "aie2");
    }

    #[test]
    fn parse_accepts_minimal_open_ledger() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).expect("minimal ledger must parse");
        assert_eq!(ledger.schema_version, 1);
        assert_eq!(ledger.tuples.len(), 1);
        assert!(ledger.inventory.is_empty());
        assert!(ledger.facts.is_empty());
        assert!(ledger.evidence.is_empty());
    }

    #[test]
    fn validation_rejects_malformed_duplicate_and_wrong_kind_ids() {
        for (pointer, replacement, expected) in [
            ("/tuples/0/id", serde_json::json!(""), "invalid tuple id"),
            ("/tuples/0/id", serde_json::json!("tuple.Test"), "invalid tuple id"),
            ("/facts/0/id", serde_json::json!("inventory.test.command"), "invalid fact id"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, replacement);
            assert_validation_error(value, expected);
        }

        let mut value = linked_value();
        let duplicate = value["tuples"][0].clone();
        value["tuples"].as_array_mut().unwrap().push(duplicate);
        assert_validation_error(value, "duplicate tuple id");
    }

    #[test]
    fn validation_reports_inventory_ids_under_the_inventory_root() {
        let mut value = linked_value();
        replace(&mut value, "/inventory/0/id", serde_json::json!("bad"));
        let json = serde_json::to_string(&value).unwrap();
        let err = ReserveLedger::from_json(&json).expect_err("malformed inventory id must fail");
        assert!(
            err.issues.iter().any(|issue| issue.path == "$.inventory[0].id"),
            "issues used the wrong JSON root: {:?}",
            err.issues
        );
    }

    #[test]
    fn validation_rejects_empty_fact_bearing_dispositions() {
        for state in ["applicable", "proven_not_applicable"] {
            let mut value = linked_value();
            replace(
                &mut value,
                "/inventory/0/disposition",
                serde_json::json!({"state": state, "fact_ids": []}),
            );
            assert_validation_error(value, "requires at least one fact");
        }
    }

    #[test]
    fn validation_rejects_dangling_references() {
        for (pointer, replacement, expected) in [
            ("/inventory/0/tuple_ids/0", serde_json::json!("tuple.missing"), "unknown tuple id"),
            ("/inventory/0/dependency_ids", serde_json::json!(["inventory.missing"]), "unknown inventory id"),
            ("/facts/0/dependency_fact_ids", serde_json::json!(["fact.missing"]), "unknown fact id"),
            (
                "/facts/0/supporting_evidence_ids/0",
                serde_json::json!("evidence.missing"),
                "unknown evidence id",
            ),
            (
                "/tuples/0/kernel_corpus_evidence_ids/0",
                serde_json::json!("evidence.missing"),
                "unknown evidence id",
            ),
            ("/evidence/0/candidate_tuple_ids/0", serde_json::json!("tuple.missing"), "unknown tuple id"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, replacement);
            assert_validation_error(value, expected);
        }
    }

    #[test]
    fn validation_rejects_unknown_coverage_domains() {
        let mut value = linked_value();
        replace(&mut value, "/inventory/0/coverage_domain_ids/0", serde_json::json!("not_a_spine_domain"));
        assert_validation_error(value, "unknown coverage domain");
    }

    #[test]
    fn validation_rejects_malformed_sha256_values() {
        for pointer in ["/tuples/0/firmware/sha256", "/evidence/0/expected_digests/manifest_sha256"] {
            let mut value = linked_value();
            replace(&mut value, pointer, serde_json::json!("ABC123"));
            assert_validation_error(value, "invalid lowercase SHA-256");
        }
    }

    #[test]
    fn validation_rejects_unsafe_external_locations() {
        for path in ["", "/absolute/path", "../escape", "safe/../escape"] {
            let mut value = linked_value();
            replace(&mut value, "/evidence/0/location/relative_path", serde_json::json!(path));
            assert_validation_error(value, "invalid relative path");
        }

        let mut value = linked_value();
        replace(&mut value, "/evidence/0/location/alias", serde_json::json!(" "));
        assert_validation_error(value, "location alias must not be blank");
    }

    #[test]
    fn validation_rejects_empty_required_text() {
        for (pointer, expected) in [
            ("/tuples/0/title", "tuple title must not be blank"),
            ("/inventory/0/title", "inventory title must not be blank"),
            ("/facts/0/statement", "fact statement must not be blank"),
            ("/facts/0/expected_transition", "expected transition must not be blank"),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, serde_json::json!("  "));
            assert_validation_error(value, expected);
        }
    }

    #[test]
    fn validation_rejects_inventory_dependency_cycles() {
        let mut self_cycle = linked_value();
        replace(
            &mut self_cycle,
            "/inventory/0/dependency_ids",
            serde_json::json!(["inventory.test.command"]),
        );
        assert_validation_error(self_cycle, "inventory dependency cycle");

        let mut two_node_cycle = linked_value();
        let mut second = two_node_cycle["inventory"][0].clone();
        second["id"] = serde_json::json!("inventory.test.second");
        second["dependency_ids"] = serde_json::json!(["inventory.test.command"]);
        second["disposition"]["fact_ids"] = serde_json::json!(["fact.test.command"]);
        two_node_cycle["inventory"].as_array_mut().unwrap().push(second);
        two_node_cycle["inventory"][0]["dependency_ids"] = serde_json::json!(["inventory.test.second"]);
        assert_validation_error(two_node_cycle, "inventory dependency cycle");
    }

    #[test]
    fn validation_rejects_fact_dependency_cycles() {
        let mut self_cycle = linked_value();
        replace(&mut self_cycle, "/facts/0/dependency_fact_ids", serde_json::json!(["fact.test.command"]));
        assert_validation_error(self_cycle, "fact dependency cycle");

        let mut two_node_cycle = linked_value();
        let mut second = two_node_cycle["facts"][0].clone();
        second["id"] = serde_json::json!("fact.test.second");
        second["dependency_fact_ids"] = serde_json::json!(["fact.test.command"]);
        two_node_cycle["facts"].as_array_mut().unwrap().push(second);
        two_node_cycle["facts"][0]["dependency_fact_ids"] = serde_json::json!(["fact.test.second"]);
        assert_validation_error(two_node_cycle, "fact dependency cycle");
    }

    #[test]
    fn release_open_tuple_identity_blocks() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).unwrap();
        let report = ledger
            .evaluate_release("tuple.test.aie2", &verified_inputs())
            .expect("known tuple must evaluate");
        assert!(has_blocker(&report, BlockerCode::TupleIdentityOpen));
    }

    #[test]
    fn release_open_inventory_scope_blocks_even_when_inventory_is_empty() {
        let ledger = ReserveLedger::from_json(MINIMAL_LEDGER).unwrap();
        let report = ledger
            .evaluate_release("tuple.test.aie2", &verified_inputs())
            .expect("known tuple must evaluate");
        assert!(has_blocker(&report, BlockerCode::InventoryScopeOpen));
    }

    #[test]
    fn release_deferred_and_unknown_inventory_entries_block() {
        for (state, code) in
            [("deferred", BlockerCode::InventoryDeferred), ("unknown", BlockerCode::InventoryUnknown)]
        {
            let mut value = linked_value();
            replace(
                &mut value,
                "/inventory/0/disposition",
                serde_json::json!({"state": state, "reason": "not closed"}),
            );
            let report = report_for(value, &verified_inputs());
            assert!(has_blocker(&report, code), "{state} entry did not block: {report:?}");
        }
    }

    #[test]
    fn release_nonqualified_fact_blocks_applicable_inventory() {
        let mut value = linked_value();
        replace(&mut value, "/facts/0/promotion", serde_json::json!({"state": "derived"}));
        let report = report_for(value, &verified_inputs());
        assert!(has_blocker(&report, BlockerCode::FactNotRetirementQualified));
        assert!(has_blocker(&report, BlockerCode::InventoryFactUnqualified));
    }

    #[test]
    fn release_legacy_evidence_blocks_verified_promotion() {
        let mut value = linked_value();
        replace(&mut value, "/facts/0/promotion", serde_json::json!({"state": "verified"}));
        replace(&mut value, "/evidence/0/expected_digests/manifest_sha256", serde_json::Value::Null);
        replace(
            &mut value,
            "/evidence/0/provenance_gaps",
            serde_json::json!(["exact emulator commit unavailable"]),
        );
        let report = report_for(value, &verified_inputs());
        assert!(has_blocker(&report, BlockerCode::EvidenceLegacyIncomplete));
        assert!(has_blocker(&report, BlockerCode::EvidenceProvenanceIncomplete));
        assert!(has_blocker(&report, BlockerCode::FactNotRetirementQualified));
    }

    #[test]
    fn release_contested_fact_blocks_the_dependent_path() {
        let mut value = linked_value();
        replace(
            &mut value,
            "/facts/0/promotion",
            serde_json::json!({
                "state": "contested",
                "reason": "new witness disagrees",
                "evidence_ids": ["evidence.test.hw"]
            }),
        );
        let mut dependent = value["facts"][0].clone();
        dependent["id"] = serde_json::json!("fact.test.dependent");
        dependent["dependency_fact_ids"] = serde_json::json!(["fact.test.command"]);
        dependent["promotion"] = serde_json::json!({"state": "retirement_qualified"});
        value["facts"].as_array_mut().unwrap().push(dependent);
        value["inventory"][0]["disposition"]["fact_ids"] = serde_json::json!(["fact.test.dependent"]);

        let report = report_for(value, &verified_inputs());
        assert!(report.blockers.iter().any(|blocker| {
            blocker.code == BlockerCode::FactContested
                && blocker.dependency_path
                    == ["inventory.test.command", "fact.test.dependent", "fact.test.command"]
        }));
    }

    #[test]
    fn release_missing_implementation_and_tests_block() {
        let mut value = linked_value();
        replace(&mut value, "/facts/0/implementation_refs", serde_json::json!([]));
        replace(&mut value, "/facts/0/test_refs", serde_json::json!([]));
        let report = report_for(value, &verified_inputs());
        assert!(has_blocker(&report, BlockerCode::ImplementationMissing));
        assert!(has_blocker(&report, BlockerCode::TestsMissing));
    }

    #[test]
    fn release_incomplete_fact_contract_blocks() {
        for (pointer, replacement, code) in [
            ("/facts/0/preconditions", serde_json::json!([]), BlockerCode::FactContractIncomplete),
            (
                "/facts/0/remaining_unknowns",
                serde_json::json!(["response payload"]),
                BlockerCode::FactUnknownsOpen,
            ),
            (
                "/facts/0/supporting_evidence_ids",
                serde_json::json!([]),
                BlockerCode::FactSupportingEvidenceMissing,
            ),
            ("/facts/0/control_evidence_ids", serde_json::json!([]), BlockerCode::FactControlEvidenceMissing),
            ("/facts/0/alternatives_ruled_out", serde_json::json!([]), BlockerCode::FactAlternativesMissing),
        ] {
            let mut value = linked_value();
            replace(&mut value, pointer, replacement);
            let report = report_for(value, &verified_inputs());
            assert!(has_blocker(&report, code), "{pointer} did not block: {report:?}");
        }
    }

    #[test]
    fn release_unaudited_evidence_blocks_every_claimed_use() {
        let inputs =
            EvaluationInputs { semantic_provenance_clean: true, evidence_audit: EvidenceAudit::default() };
        let report = linked_ledger()
            .evaluate_release("tuple.test.aie2", &inputs)
            .expect("known tuple must evaluate");

        assert!(has_blocker(&report, BlockerCode::EvidenceUnaudited));
        assert!(has_blocker(&report, BlockerCode::LiveAttestationUnaudited));
        assert!(has_blocker(&report, BlockerCode::OfflineRehearsalUnaudited));
    }

    #[test]
    fn release_requires_two_verified_witness_replicas() {
        let inputs = EvaluationInputs {
            semantic_provenance_clean: true,
            evidence_audit: EvidenceAudit {
                verified_evidence_ids: BTreeSet::from(["evidence.test.hw".into()]),
                verified_replica_ids: BTreeSet::from([(
                    "evidence.test.hw".into(),
                    "replica.test.one".into(),
                )]),
            },
        };
        let report = linked_ledger()
            .evaluate_release("tuple.test.aie2", &inputs)
            .expect("known tuple must evaluate");
        assert!(has_blocker(&report, BlockerCode::ReplicaInsufficient));
    }

    #[test]
    fn release_missing_attestation_and_rehearsal_block() {
        let mut value = linked_value();
        replace(&mut value, "/tuples/0/live_attestation_evidence_ids", serde_json::json!([]));
        replace(&mut value, "/tuples/0/offline_rehearsal_evidence_ids", serde_json::json!([]));
        let report = report_for(value, &verified_inputs());
        assert!(has_blocker(&report, BlockerCode::LiveAttestationMissing));
        assert!(has_blocker(&report, BlockerCode::OfflineRehearsalMissing));
    }

    #[test]
    fn release_semantic_provenance_gap_blocks() {
        let inputs = EvaluationInputs { semantic_provenance_clean: false, ..verified_inputs() };
        let report = linked_ledger()
            .evaluate_release("tuple.test.aie2", &inputs)
            .expect("known tuple must evaluate");
        assert!(has_blocker(&report, BlockerCode::SemanticProvenanceOpen));
    }

    #[test]
    fn release_fully_closed_synthetic_ledger_is_clean() {
        let report = linked_ledger()
            .evaluate_release("tuple.test.aie2", &verified_inputs())
            .expect("known tuple must evaluate");
        assert!(report.blockers.is_empty(), "unexpected blockers: {:?}", report.blockers);
        assert!(report.checks.iter().all(|check| check.passed));
        assert!(report.is_clean);
    }

    #[test]
    fn release_production_gate_cannot_trust_ledger_evidence() {
        let report = linked_ledger()
            .clean_release("tuple.test.aie2")
            .expect("known tuple must evaluate");
        assert!(has_blocker(&report, BlockerCode::EvidenceUnaudited));
        assert!(has_blocker(&report, BlockerCode::ReplicaInsufficient));
        assert!(!report.is_clean);
    }

    #[test]
    fn release_unknown_tuple_is_an_error() {
        let err = linked_ledger()
            .evaluate_release("tuple.missing", &verified_inputs())
            .expect_err("unknown tuple must not produce a synthetic report");
        assert!(err.to_string().contains("unknown tuple id `tuple.missing`"));
    }

    #[test]
    fn embedded_npu1_ledger_has_stable_chain_and_pins() {
        let ledger = ReserveLedger::npu1().expect("embedded NPU1 ledger must validate");
        assert_eq!(
            ledger.tuples.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(),
            ["tuple.npu1.phoenix.fw-1_5_5_391"]
        );
        assert_eq!(
            ledger.inventory.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(),
            ["inventory.npu1.firmware.command-list-execution"]
        );
        assert_eq!(
            ledger.facts.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(),
            ["fact.npu1.firmware.command-list-lifecycle-candidate"]
        );
        assert_eq!(
            ledger.evidence.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(),
            ["evidence.npu1.legacy-vfio-user-chess-20260729t171244z"]
        );

        let tuple = &ledger.tuples[0];
        assert_eq!(tuple.architecture, Architecture::Aie2);
        assert_eq!((tuple.device.vendor_id.as_str(), tuple.device.device_id.as_str()), ("1022", "1502"));
        assert_eq!(tuple.firmware.logical_name, "amdnpu/1502_00/npu.dev.sbin");
        assert_eq!(tuple.firmware.sha256, "d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e");
        assert_eq!(tuple.driver_surface.commit, "216cefececd74effcd7a88350c71b99f5ef9a215");

        let evidence = &ledger.evidence[0];
        assert_eq!(evidence.kind, EvidenceKind::HistoricalEmulatorWitness);
        assert_eq!(evidence.location.alias, "repo-experiments");
        assert_eq!(evidence.location.relative_path, "phoenix-vfio-user/20260729T171244Z-3136359");
        assert_eq!(
            evidence.expected_digests.metadata_fingerprint_sha256.as_deref(),
            Some("4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299")
        );
        assert_eq!(
            evidence.expected_digests.checksum_index_sha256.as_deref(),
            Some("e7aaacefa4c8f3606529dd27980397a656b22099a349db59d1c0df84330811e2")
        );
        assert!(evidence.expected_digests.manifest_sha256.is_none());
        assert!(evidence.expected_replicas.is_empty());
    }

    #[test]
    fn embedded_npu1_release_blocker_codes_are_exact() {
        let ledger = ReserveLedger::npu1().expect("embedded NPU1 ledger must validate");
        let report = ledger
            .clean_release("tuple.npu1.phoenix.fw-1_5_5_391")
            .expect("primary tuple must evaluate");
        let actual = report.blockers.iter().map(|blocker| blocker.code).collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            BlockerCode::TupleIdentityOpen,
            BlockerCode::InventoryScopeOpen,
            BlockerCode::InventoryFactUnqualified,
            BlockerCode::FactNotRetirementQualified,
            BlockerCode::FactUnknownsOpen,
            BlockerCode::FactControlEvidenceMissing,
            BlockerCode::FactAlternativesMissing,
            BlockerCode::ImplementationMissing,
            BlockerCode::TestsMissing,
            BlockerCode::EvidenceLegacyIncomplete,
            BlockerCode::EvidenceProvenanceIncomplete,
            BlockerCode::EvidenceUnaudited,
            BlockerCode::ReplicaInsufficient,
            BlockerCode::SemanticProvenanceOpen,
            BlockerCode::LiveAttestationMissing,
            BlockerCode::OfflineRehearsalMissing,
        ]);
        assert_eq!(actual, expected);
        assert!(!report.is_clean);
    }

    #[test]
    fn render_release_report_is_deterministic_across_record_order() {
        let ledger = ReserveLedger::npu1().unwrap();
        let mut value = serde_json::to_value(&ledger).unwrap();

        let mut fact = value["facts"][0].clone();
        fact["id"] = serde_json::json!("fact.npu1.firmware.a-second");
        value["facts"].as_array_mut().unwrap().push(fact);

        let mut inventory = value["inventory"][0].clone();
        inventory["id"] = serde_json::json!("inventory.npu1.firmware.a-second");
        inventory["disposition"]["fact_ids"] = serde_json::json!(["fact.npu1.firmware.a-second"]);
        value["inventory"].as_array_mut().unwrap().push(inventory);

        let json = serde_json::to_string(&value).unwrap();
        let ordered = ReserveLedger::from_json(&json).expect("two-chain ledger must validate");
        let ordered_report = ordered.clean_release("tuple.npu1.phoenix.fw-1_5_5_391").unwrap();
        let expected = render_release_report(&ordered, &ordered_report);

        let mut reversed = ordered;
        reversed.inventory.reverse();
        reversed.facts.reverse();
        let reversed_report = reversed.clean_release("tuple.npu1.phoenix.fw-1_5_5_391").unwrap();
        assert_eq!(render_release_report(&reversed, &reversed_report), expected);
    }

    #[test]
    fn render_release_report_contains_contract_and_warning() {
        let ledger = ReserveLedger::npu1().unwrap();
        let report = ledger.clean_release("tuple.npu1.phoenix.fw-1_5_5_391").unwrap();
        let rendered = render_release_report(&ledger, &report);

        for check in [
            "tuple_identity",
            "inventory",
            "fact",
            "implementation",
            "evidence",
            "replica",
            "semantic_provenance",
            "live_attestation",
            "offline_rehearsal",
        ] {
            assert!(rendered.contains(&format!("| `{check}` |")), "missing `{check}` check");
        }
        for required in [
            "amdnpu/1502_00/npu.dev.sbin",
            "d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e",
            "216cefececd74effcd7a88350c71b99f5ef9a215",
            "4d80663aecf902e12c46fac3fcca95955a5ee04a1ba4aaf0397354dcd52d2299",
            "e7aaacefa4c8f3606529dd27980397a656b22099a349db59d1c0df84330811e2",
            "exact board identity",
            "complete amdxdna driver surface census",
            "inventory.npu1.firmware.command-list-execution -> fact.npu1.firmware.command-list-lifecycle-candidate",
            "Historical emulator witnesses are regression evidence only",
            "not physical NPU evidence",
            "npu1-research-reserve/snapshots/2026-07-29-pre-slice-b",
        ] {
            assert!(rendered.contains(required), "missing `{required}`");
        }
    }

    #[test]
    fn release_report_is_not_stale() {
        let ledger = ReserveLedger::npu1().unwrap();
        let report = ledger.clean_release("tuple.npu1.phoenix.fw-1_5_5_391").unwrap();
        let want = render_release_report(&ledger, &report);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("docs/coverage/npu1/release-report.md");
        let got = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            got,
            want,
            "{} is stale -- regenerate: `cargo run -p xdna-archspec --example gen_coverage_artifacts`",
            path.display()
        );
    }
}
