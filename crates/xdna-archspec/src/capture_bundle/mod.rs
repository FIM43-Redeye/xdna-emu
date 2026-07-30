use crate::{
    research_reserve::{ContentPin, Redistributability, RevisionPin},
    types::Architecture,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::PathBuf,
};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const EMISSION_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub campaign: Campaign,
    pub artifacts: Vec<ArtifactRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmissionPlan {
    pub schema_version: u32,
    pub campaign: Campaign,
    pub artifacts: Vec<ArtifactSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Campaign {
    pub id: String,
    pub tuple_ids: Vec<String>,
    pub inventory_ids: Vec<String>,
    pub fact_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub provenance: Provenance,
    pub risk_class: String,
    pub outcome: CampaignOutcome,
    pub platform: PlatformIdentity,
    pub stimulus: Stimulus,
    pub runs: Vec<RunRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    Current,
    Legacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignOutcome {
    Success,
    IntentionalRejection,
    InfrastructureFailure,
    ProvenanceFailure,
    SemanticMismatch,
    DeviceFaultOrWedge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Availability<T> {
    Known { value: T },
    Unavailable { reason: String },
    NotApplicable { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformIdentity {
    #[serde(with = "architecture_serde")]
    pub architecture: Architecture,
    pub device_model_key: Availability<String>,
    pub driver_platform_id: Availability<String>,
    pub pci: Availability<PciIdentity>,
    pub board_identity: Availability<String>,
    pub firmware: Availability<ContentPin>,
    pub host_kernel: Availability<ComponentPin>,
    pub kernel_modules: Vec<ComponentPin>,
    pub driver: Availability<RevisionPin>,
    pub xrt_components: Vec<ComponentPin>,
    pub toolchain_components: Vec<ComponentPin>,
    pub compiler_mode: Availability<String>,
    pub execution_mode: Availability<String>,
    pub reset_state: Availability<String>,
    pub power_state: Availability<String>,
    pub clock_state: Availability<String>,
    pub iommu_state: Availability<String>,
    pub address_state: Availability<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PciIdentity {
    pub vendor_id: String,
    pub device_id: String,
    pub subsystem_vendor_id: String,
    pub subsystem_device_id: String,
    pub revision_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentPin {
    pub name: String,
    pub revision: String,
    pub sha256: Availability<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stimulus {
    pub command: CommandStimulus,
    pub source_revisions: Vec<RevisionPin>,
    pub build_recipe: Availability<ContentPin>,
    pub inputs: Vec<InputIdentity>,
    pub initial_state: Vec<String>,
    pub external_events: Vec<ExternalEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandStimulus {
    pub argv: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputIdentity {
    pub id: String,
    pub semantic_kind: String,
    pub content: ContentPin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalEvent {
    pub id: String,
    pub ordinal: u32,
    pub description: String,
    pub offset: ExactTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactTime {
    pub value: u64,
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub id: String,
    pub ordinal: u32,
    pub repetition: u32,
    pub completion: String,
    pub output_artifact_paths: Vec<String>,
    pub observations: Vec<ObservationRecord>,
    pub timing: Vec<TimingBound>,
    pub errors: Vec<String>,
    pub recovery_actions: Vec<String>,
    pub teardown: String,
    pub control_run_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRecord {
    pub id: String,
    pub semantic_kind: String,
    pub artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingBound {
    pub anchor: String,
    pub lower: ExactTime,
    pub upper: ExactTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactClass {
    Raw,
    Derived,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecord {
    pub path: String,
    pub byte_size: u64,
    pub sha256: String,
    pub semantic_kind: String,
    pub class: ArtifactClass,
    pub redistributability: Redistributability,
    pub run_ids: Vec<String>,
    pub observation_ids: Vec<String>,
    pub derivation: Option<DerivationProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSource {
    pub source_path: PathBuf,
    pub path: String,
    pub semantic_kind: String,
    pub class: ArtifactClass,
    pub redistributability: Redistributability,
    pub run_ids: Vec<String>,
    pub observation_ids: Vec<String>,
    pub derivation: Option<DerivationProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivationProvenance {
    pub source_artifact_paths: Vec<String>,
    pub source_bundle_ids: Vec<String>,
    pub command: CommandStimulus,
    pub analysis_tool: RevisionPin,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BundleIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleSchemaError {
    issues: Vec<BundleIssue>,
}

impl BundleSchemaError {
    pub fn issues(&self) -> &[BundleIssue] {
        &self.issues
    }
}

impl fmt::Display for BundleSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capture bundle schema validation failed")?;
        for issue in &self.issues {
            write!(f, "; {}: {}", issue.path, issue.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for BundleSchemaError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionEligibility {
    blockers: Vec<BundleIssue>,
}

impl PromotionEligibility {
    pub fn is_promotion_eligible(&self) -> bool {
        self.blockers.is_empty()
    }

    pub fn blockers(&self) -> &[BundleIssue] {
        &self.blockers
    }
}

impl BundleManifest {
    pub fn validate(&self) -> Result<PromotionEligibility, BundleSchemaError> {
        let mut issues = Vec::new();
        let mut blockers = Vec::new();
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            issue(
                &mut issues,
                "$.schema_version",
                format!("unsupported schema_version {}", self.schema_version),
            );
        }
        validate_bundle_id(&self.bundle_id, "$.bundle_id", &mut issues);
        let artifact_paths = validate_artifact_records(&self.artifacts, &self.campaign, &mut issues);
        validate_campaign(&self.campaign, &artifact_paths, &mut issues, &mut blockers);
        finish_validation(issues, blockers)
    }
}

impl EmissionPlan {
    pub fn validate(&self) -> Result<PromotionEligibility, BundleSchemaError> {
        let mut issues = Vec::new();
        let mut blockers = Vec::new();
        if self.schema_version != EMISSION_PLAN_SCHEMA_VERSION {
            issue(
                &mut issues,
                "$.schema_version",
                format!("unsupported schema_version {}", self.schema_version),
            );
        }
        let artifact_paths = validate_artifact_sources(&self.artifacts, &self.campaign, &mut issues);
        validate_campaign(&self.campaign, &artifact_paths, &mut issues, &mut blockers);
        finish_validation(issues, blockers)
    }
}

fn finish_validation(
    mut issues: Vec<BundleIssue>,
    mut blockers: Vec<BundleIssue>,
) -> Result<PromotionEligibility, BundleSchemaError> {
    issues.sort();
    issues.dedup();
    blockers.sort();
    blockers.dedup();
    if issues.is_empty() {
        Ok(PromotionEligibility { blockers })
    } else {
        Err(BundleSchemaError { issues })
    }
}

fn validate_campaign(
    campaign: &Campaign,
    artifact_paths: &BTreeSet<String>,
    issues: &mut Vec<BundleIssue>,
    blockers: &mut Vec<BundleIssue>,
) {
    validate_stable_string(&campaign.id, "$.campaign.id", "campaign ID", issues);
    let mut ledger_ids = BTreeSet::new();
    validate_id_list(&campaign.tuple_ids, "$.campaign.tuple_ids", &mut ledger_ids, issues);
    validate_id_list(&campaign.inventory_ids, "$.campaign.inventory_ids", &mut ledger_ids, issues);
    validate_id_list(&campaign.fact_ids, "$.campaign.fact_ids", &mut ledger_ids, issues);
    validate_id_list(&campaign.evidence_ids, "$.campaign.evidence_ids", &mut ledger_ids, issues);
    if campaign.provenance == Provenance::Legacy {
        blocker(blockers, "$.campaign.provenance", "legacy provenance is not promotion-eligible");
    }
    validate_stable_string(&campaign.risk_class, "$.campaign.risk_class", "risk class", issues);
    validate_platform(&campaign.platform, issues, blockers);
    validate_stimulus(&campaign.stimulus, issues, blockers);
    validate_runs(&campaign.runs, artifact_paths, issues);
}

fn validate_platform(
    platform: &PlatformIdentity,
    issues: &mut Vec<BundleIssue>,
    blockers: &mut Vec<BundleIssue>,
) {
    validate_available(
        &platform.device_model_key,
        "$.campaign.platform.device_model_key",
        false,
        issues,
        blockers,
        |value, path, issues| validate_stable_string(value, path, "device-model key", issues),
    );
    validate_available(
        &platform.driver_platform_id,
        "$.campaign.platform.driver_platform_id",
        false,
        issues,
        blockers,
        |value, path, issues| validate_stable_string(value, path, "driver platform ID", issues),
    );
    validate_available(&platform.pci, "$.campaign.platform.pci", false, issues, blockers, validate_pci);
    validate_available(
        &platform.board_identity,
        "$.campaign.platform.board_identity",
        true,
        issues,
        blockers,
        |value, path, issues| require_text(value, path, "board identity", issues),
    );
    validate_available(
        &platform.firmware,
        "$.campaign.platform.firmware",
        false,
        issues,
        blockers,
        validate_content_pin,
    );
    validate_available(
        &platform.host_kernel,
        "$.campaign.platform.host_kernel",
        false,
        issues,
        blockers,
        validate_component,
    );
    if let Availability::Known { value } = &platform.host_kernel {
        validate_available(
            &value.sha256,
            "$.campaign.platform.host_kernel.sha256",
            false,
            issues,
            blockers,
            |value, path, issues| validate_sha256(value, path, issues),
        );
    }
    validate_components(&platform.kernel_modules, "$.campaign.platform.kernel_modules", issues, blockers);
    validate_available(
        &platform.driver,
        "$.campaign.platform.driver",
        false,
        issues,
        blockers,
        validate_revision_pin,
    );
    validate_components(&platform.xrt_components, "$.campaign.platform.xrt_components", issues, blockers);
    validate_components(
        &platform.toolchain_components,
        "$.campaign.platform.toolchain_components",
        issues,
        blockers,
    );
    for (name, value) in [
        ("compiler_mode", &platform.compiler_mode),
        ("execution_mode", &platform.execution_mode),
        ("reset_state", &platform.reset_state),
        ("power_state", &platform.power_state),
        ("clock_state", &platform.clock_state),
        ("iommu_state", &platform.iommu_state),
        ("address_state", &platform.address_state),
    ] {
        let path = format!("$.campaign.platform.{name}");
        validate_available(value, &path, false, issues, blockers, |value, path, issues| {
            require_text(value, path, "platform state", issues)
        });
    }
}

fn validate_pci(pci: &PciIdentity, path: &str, issues: &mut Vec<BundleIssue>) {
    for (name, value) in [
        ("vendor_id", &pci.vendor_id),
        ("device_id", &pci.device_id),
        ("subsystem_vendor_id", &pci.subsystem_vendor_id),
        ("subsystem_device_id", &pci.subsystem_device_id),
        ("revision_id", &pci.revision_id),
    ] {
        require_text(value, &format!("{path}.{name}"), "PCI identity", issues);
    }
}

fn validate_components(
    components: &[ComponentPin],
    path: &str,
    issues: &mut Vec<BundleIssue>,
    blockers: &mut Vec<BundleIssue>,
) {
    let mut names = BTreeSet::new();
    for (index, component) in components.iter().enumerate() {
        let component_path = format!("{path}[{index}]");
        validate_component(component, &component_path, issues);
        validate_available(
            &component.sha256,
            &format!("{component_path}.sha256"),
            false,
            issues,
            blockers,
            |value, path, issues| validate_sha256(value, path, issues),
        );
        if !names.insert(&component.name) {
            issue(
                issues,
                format!("{component_path}.name"),
                format!("duplicate component name `{}`", component.name),
            );
        }
    }
}

fn validate_component(component: &ComponentPin, path: &str, issues: &mut Vec<BundleIssue>) {
    require_text(&component.name, &format!("{path}.name"), "component name", issues);
    require_text(&component.revision, &format!("{path}.revision"), "component revision", issues);
}

fn validate_stimulus(stimulus: &Stimulus, issues: &mut Vec<BundleIssue>, blockers: &mut Vec<BundleIssue>) {
    validate_command(&stimulus.command, "$.campaign.stimulus.command", issues);
    for (index, revision) in stimulus.source_revisions.iter().enumerate() {
        validate_revision_pin(revision, &format!("$.campaign.stimulus.source_revisions[{index}]"), issues);
    }
    validate_available(
        &stimulus.build_recipe,
        "$.campaign.stimulus.build_recipe",
        false,
        issues,
        blockers,
        validate_content_pin,
    );
    let mut input_ids = BTreeSet::new();
    for (index, input) in stimulus.inputs.iter().enumerate() {
        let path = format!("$.campaign.stimulus.inputs[{index}]");
        validate_id(&input.id, &format!("{path}.id"), &mut input_ids, issues);
        validate_stable_string(
            &input.semantic_kind,
            &format!("{path}.semantic_kind"),
            "input semantic kind",
            issues,
        );
        validate_content_pin(&input.content, &format!("{path}.content"), issues);
    }
    validate_text_list(&stimulus.initial_state, "$.campaign.stimulus.initial_state", issues);
    let mut event_ids = BTreeSet::new();
    for (index, event) in stimulus.external_events.iter().enumerate() {
        let path = format!("$.campaign.stimulus.external_events[{index}]");
        validate_id(&event.id, &format!("{path}.id"), &mut event_ids, issues);
        require_text(
            &event.description,
            &format!("{path}.description"),
            "external-event description",
            issues,
        );
        validate_exact_time(&event.offset, &format!("{path}.offset"), issues);
    }
}

fn validate_runs(runs: &[RunRecord], artifact_paths: &BTreeSet<String>, issues: &mut Vec<BundleIssue>) {
    let mut run_ids = BTreeSet::new();
    let mut observation_ids = BTreeSet::new();
    for (index, run) in runs.iter().enumerate() {
        let path = format!("$.campaign.runs[{index}]");
        validate_id(&run.id, &format!("{path}.id"), &mut run_ids, issues);
        validate_stable_string(&run.completion, &format!("{path}.completion"), "run completion", issues);
        validate_artifact_refs(
            &run.output_artifact_paths,
            artifact_paths,
            &format!("{path}.output_artifact_paths"),
            issues,
        );
        for (observation_index, observation) in run.observations.iter().enumerate() {
            let observation_path = format!("{path}.observations[{observation_index}]");
            validate_id(&observation.id, &format!("{observation_path}.id"), &mut observation_ids, issues);
            validate_stable_string(
                &observation.semantic_kind,
                &format!("{observation_path}.semantic_kind"),
                "observation semantic kind",
                issues,
            );
            validate_artifact_refs(
                &observation.artifact_paths,
                artifact_paths,
                &format!("{observation_path}.artifact_paths"),
                issues,
            );
        }
        for (timing_index, timing) in run.timing.iter().enumerate() {
            let timing_path = format!("{path}.timing[{timing_index}]");
            validate_stable_string(&timing.anchor, &format!("{timing_path}.anchor"), "timing anchor", issues);
            validate_exact_time(&timing.lower, &format!("{timing_path}.lower"), issues);
            validate_exact_time(&timing.upper, &format!("{timing_path}.upper"), issues);
            if timing.lower.unit != timing.upper.unit {
                issue(issues, format!("{timing_path}.upper.unit"), "timing-bound units must match");
            } else if timing.lower.value > timing.upper.value {
                issue(
                    issues,
                    format!("{timing_path}.upper.value"),
                    "timing upper bound precedes lower bound",
                );
            }
        }
        validate_text_list(&run.errors, &format!("{path}.errors"), issues);
        validate_text_list(&run.recovery_actions, &format!("{path}.recovery_actions"), issues);
        validate_stable_string(&run.teardown, &format!("{path}.teardown"), "teardown result", issues);
    }

    for (index, run) in runs.iter().enumerate() {
        validate_refs(
            &run.control_run_ids,
            &run_ids,
            &format!("$.campaign.runs[{index}].control_run_ids"),
            "run",
            issues,
        );
    }
}

fn validate_artifact_records(
    artifacts: &[ArtifactRecord],
    campaign: &Campaign,
    issues: &mut Vec<BundleIssue>,
) -> BTreeSet<String> {
    let paths: BTreeSet<String> = artifacts.iter().map(|artifact| artifact.path.clone()).collect();
    let (run_ids, observation_ids) = campaign_reference_ids(campaign);
    let mut seen = BTreeSet::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let path = format!("$.artifacts[{index}]");
        validate_artifact_metadata(
            &artifact.path,
            &artifact.semantic_kind,
            artifact.class,
            &artifact.run_ids,
            &artifact.observation_ids,
            artifact.derivation.as_ref(),
            &path,
            &paths,
            &run_ids,
            &observation_ids,
            &mut seen,
            issues,
        );
        validate_sha256(&artifact.sha256, &format!("{path}.sha256"), issues);
    }
    paths
}

fn validate_artifact_sources(
    artifacts: &[ArtifactSource],
    campaign: &Campaign,
    issues: &mut Vec<BundleIssue>,
) -> BTreeSet<String> {
    let paths: BTreeSet<String> = artifacts.iter().map(|artifact| artifact.path.clone()).collect();
    let (run_ids, observation_ids) = campaign_reference_ids(campaign);
    let mut seen = BTreeSet::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let path = format!("$.artifacts[{index}]");
        if artifact.source_path.as_os_str().is_empty() {
            issue(issues, format!("{path}.source_path"), "source path must not be empty");
        }
        validate_artifact_metadata(
            &artifact.path,
            &artifact.semantic_kind,
            artifact.class,
            &artifact.run_ids,
            &artifact.observation_ids,
            artifact.derivation.as_ref(),
            &path,
            &paths,
            &run_ids,
            &observation_ids,
            &mut seen,
            issues,
        );
    }
    paths
}

fn campaign_reference_ids(campaign: &Campaign) -> (BTreeSet<String>, BTreeSet<String>) {
    (
        campaign.runs.iter().map(|run| run.id.clone()).collect(),
        campaign
            .runs
            .iter()
            .flat_map(|run| run.observations.iter())
            .map(|observation| observation.id.clone())
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_artifact_metadata(
    artifact_path: &str,
    semantic_kind: &str,
    class: ArtifactClass,
    run_ids: &[String],
    observation_ids: &[String],
    derivation: Option<&DerivationProvenance>,
    path: &str,
    artifact_paths: &BTreeSet<String>,
    known_run_ids: &BTreeSet<String>,
    known_observation_ids: &BTreeSet<String>,
    seen: &mut BTreeSet<String>,
    issues: &mut Vec<BundleIssue>,
) {
    validate_artifact_path(artifact_path, class, &format!("{path}.path"), issues);
    if !seen.insert(artifact_path.to_owned()) {
        issue(issues, format!("{path}.path"), format!("duplicate artifact path `{artifact_path}`"));
    }
    validate_stable_string(semantic_kind, &format!("{path}.semantic_kind"), "artifact semantic kind", issues);
    validate_refs(run_ids, known_run_ids, &format!("{path}.run_ids"), "run", issues);
    validate_refs(
        observation_ids,
        known_observation_ids,
        &format!("{path}.observation_ids"),
        "observation",
        issues,
    );
    match (class, derivation) {
        (ArtifactClass::Raw, Some(_)) => {
            issue(issues, format!("{path}.derivation"), "raw artifact must not claim derivation")
        }
        (ArtifactClass::Derived, None) => {
            issue(issues, format!("{path}.derivation"), "derived artifact requires derivation provenance")
        }
        (_, Some(derivation)) => {
            validate_artifact_refs(
                &derivation.source_artifact_paths,
                artifact_paths,
                &format!("{path}.derivation.source_artifact_paths"),
                issues,
            );
            for (index, bundle_id) in derivation.source_bundle_ids.iter().enumerate() {
                validate_bundle_id(
                    bundle_id,
                    &format!("{path}.derivation.source_bundle_ids[{index}]"),
                    issues,
                );
            }
            validate_command(&derivation.command, &format!("{path}.derivation.command"), issues);
            validate_revision_pin(
                &derivation.analysis_tool,
                &format!("{path}.derivation.analysis_tool"),
                issues,
            );
        }
        _ => {}
    }
}

fn validate_available<T>(
    availability: &Availability<T>,
    path: &str,
    not_applicable_allowed: bool,
    issues: &mut Vec<BundleIssue>,
    blockers: &mut Vec<BundleIssue>,
    validate_known: impl FnOnce(&T, &str, &mut Vec<BundleIssue>),
) {
    match availability {
        Availability::Known { value } => validate_known(value, path, issues),
        Availability::Unavailable { reason } => {
            require_text(reason, path, "unavailable reason", issues);
            blocker(blockers, path, format!("identity unavailable: {reason}"));
        }
        Availability::NotApplicable { reason } => {
            require_text(reason, path, "not-applicable reason", issues);
            if !not_applicable_allowed {
                issue(issues, path, "not_applicable is not permitted for this field");
            }
        }
    }
}

fn validate_command(command: &CommandStimulus, path: &str, issues: &mut Vec<BundleIssue>) {
    if command.argv.is_empty() {
        issue(issues, format!("{path}.argv"), "command argv must not be empty");
    }
    validate_text_list(&command.argv, &format!("{path}.argv"), issues);
    for name in command.environment.keys() {
        require_text(name, &format!("{path}.environment"), "environment name", issues);
    }
}

fn validate_content_pin(pin: &ContentPin, path: &str, issues: &mut Vec<BundleIssue>) {
    require_text(&pin.logical_name, &format!("{path}.logical_name"), "content logical name", issues);
    validate_sha256(&pin.sha256, &format!("{path}.sha256"), issues);
}

fn validate_revision_pin(pin: &RevisionPin, path: &str, issues: &mut Vec<BundleIssue>) {
    require_text(&pin.repository, &format!("{path}.repository"), "revision repository", issues);
    require_text(&pin.commit, &format!("{path}.commit"), "revision commit", issues);
}

fn validate_exact_time(time: &ExactTime, path: &str, issues: &mut Vec<BundleIssue>) {
    validate_stable_string(&time.unit, &format!("{path}.unit"), "time unit", issues);
}

fn validate_artifact_path(value: &str, class: ArtifactClass, path: &str, issues: &mut Vec<BundleIssue>) {
    let components: Vec<&str> = value.split('/').collect();
    let expected_root = match class {
        ArtifactClass::Raw => "raw",
        ArtifactClass::Derived => "derived",
    };
    let invalid = components.len() < 2
        || components[0] != expected_root
        || components
            .iter()
            .any(|component| component.is_empty() || matches!(*component, "." | ".."))
        || value.contains('\\')
        || value.chars().any(char::is_control);
    if invalid {
        issue(issues, path, format!("invalid canonical {expected_root} artifact path `{value}`"));
    }
}

fn validate_artifact_refs(
    refs: &[String],
    known: &BTreeSet<String>,
    path: &str,
    issues: &mut Vec<BundleIssue>,
) {
    validate_refs(refs, known, path, "artifact", issues);
}

fn validate_refs(
    refs: &[String],
    known: &BTreeSet<String>,
    path: &str,
    kind: &str,
    issues: &mut Vec<BundleIssue>,
) {
    let mut seen = BTreeSet::new();
    for (index, reference) in refs.iter().enumerate() {
        let reference_path = format!("{path}[{index}]");
        if !seen.insert(reference) {
            issue(issues, &reference_path, format!("duplicate {kind} reference `{reference}`"));
        }
        if !known.contains(reference) {
            issue(issues, reference_path, format!("unknown {kind} reference `{reference}`"));
        }
    }
}

fn validate_id_list(ids: &[String], path: &str, seen: &mut BTreeSet<String>, issues: &mut Vec<BundleIssue>) {
    for (index, id) in ids.iter().enumerate() {
        validate_id(id, &format!("{path}[{index}]"), seen, issues);
    }
}

fn validate_id(id: &str, path: &str, seen: &mut BTreeSet<String>, issues: &mut Vec<BundleIssue>) {
    validate_stable_string(id, path, "stable ID", issues);
    if !seen.insert(id.to_owned()) {
        issue(issues, path, format!("duplicate stable ID `{id}`"));
    }
}

fn validate_stable_string(value: &str, path: &str, label: &str, issues: &mut Vec<BundleIssue>) {
    if !valid_dotted_id(value) {
        issue(issues, path, format!("invalid {label} `{value}`"));
    }
}

fn valid_dotted_id(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            matches!(bytes.next(), Some(first) if first.is_ascii_lowercase() || first.is_ascii_digit())
                && bytes.all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
                })
        })
}

fn validate_bundle_id(value: &str, path: &str, issues: &mut Vec<BundleIssue>) {
    match value.strip_prefix("bundle.sha256.") {
        Some(hash) => validate_sha256(hash, path, issues),
        None => issue(issues, path, format!("invalid bundle ID `{value}`")),
    }
}

fn validate_sha256(value: &str, path: &str, issues: &mut Vec<BundleIssue>) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')) {
        issue(issues, path, format!("invalid lowercase SHA-256 `{value}`"));
    }
}

fn validate_text_list(values: &[String], path: &str, issues: &mut Vec<BundleIssue>) {
    for (index, value) in values.iter().enumerate() {
        require_text(value, &format!("{path}[{index}]"), "list entry", issues);
    }
}

fn require_text(value: &str, path: &str, label: &str, issues: &mut Vec<BundleIssue>) {
    if value.trim().is_empty() {
        issue(issues, path, format!("{label} must not be blank"));
    }
}

fn issue(issues: &mut Vec<BundleIssue>, path: impl Into<String>, message: impl Into<String>) {
    issues.push(BundleIssue { path: path.into(), message: message.into() });
}

fn blocker(blockers: &mut Vec<BundleIssue>, path: impl Into<String>, message: impl Into<String>) {
    blockers.push(BundleIssue { path: path.into(), message: message.into() });
}

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

mod canonical;
pub use canonical::{build_canonical_bundle, canonicalize_manifest, CanonicalBundle};

#[cfg(test)]
mod tests;
