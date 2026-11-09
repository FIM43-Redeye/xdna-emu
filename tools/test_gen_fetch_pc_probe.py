"""The mode-1 fetch-PC probe: case table, kernel text, and the site bookkeeping."""
import gen_fetch_pc_probe as g


def test_every_case_preserves_the_aiecc_mains_registers_and_the_link_register():
    for name, case in g.cases().items():
        asm = case.consumer_asm()
        assert "st r16, [sp, #-4]" in asm and "lda r16, [sp, #-4]" in asm, name
        assert "st r17, [sp, #-8]" in asm and "lda r17, [sp, #-8]" in asm, name
        # lr survives the kernel's own calls: saved in r10, put back before the epilogue's ret
        assert asm.index("mov r10, lr") < asm.index("mov lr, r10") < asm.rindex("ret lr"), name


def test_every_site_label_is_defined_exactly_once():
    for name, case in g.cases().items():
        asm = case.consumer_asm()
        for _event, label in case.sites:
            assert asm.count(f"\n{label}:") == 1, (name, label)


def test_branch_variants_put_the_event_at_every_bundle_around_the_branch():
    body, sites, callees = g.branch_variants("jl")
    positions = sorted(int(label.split("_c")[-1]) if "_c" in label else -1 for _e, label in sites if "_c" in label)
    assert positions == [p + 5 for p in range(-5, 6) if p != 0]
    assert any("ret lr" in line for line in callees)


def test_loop_variants_run_each_loop_four_times():
    _body, sites = g.loop_variants()
    assert len(sites) == 7 * 4
    assert sites[0] == sites[3] != sites[4]


def test_where_reads_bundles_after_the_nearest_site():
    bundles = [(0x100, "nop"), (0x102, "nop"), (0x106, "event #0")]
    assert g.where(0x106, bundles, {"site_a": 0x100}) == "site_a+2b (0x106)"
