"""Paper Table 4 hierarchy shared by the ASAP7 report and result parser."""

from __future__ import annotations


REPORT_COMPONENTS = {
    "mvu": (
        "mmcu",
        "mmcu/*",
        "vectorALU",
        "vectorALU/*",
        "bilinear",
        "bilinear/*",
        "normUnit",
        "normUnit/*",
        "activation",
        "activation/*",
    ),
    "mvu_mmcu": ("mmcu", "mmcu/*"),
    "mvu_vector_alu": ("vectorALU", "vectorALU/*"),
    "mvu_bilinear_unit": ("bilinear", "bilinear/*"),
    "mvu_norm_unit": ("normUnit", "normUnit/*"),
    "mvu_activation_unit": ("activation", "activation/*"),
    "ggu_array": ("gguArray", "gguArray/*"),
    "ggu_position_calc": ("gguArray/pes_*/posCalc", "gguArray/pes_*/posCalc/*"),
    "ggu_cov_builder": ("gguArray/pes_*/covBld", "gguArray/pes_*/covBld/*"),
    "ggu_sh_op_generator": ("gguArray/pes_*/shOpGen", "gguArray/pes_*/shOpGen/*"),
    "fsdr_subsystem": (
        "fsdrCtrl",
        "fsdrCtrl/*",
        "lshHash",
        "lshHash/*",
        "fsdrCache",
        "fsdrCache/*",
    ),
    "fsdr_lsh_hash_unit": ("lshHash", "lshHash/*"),
    "fsdr_cam_array": ("fsdrCache", "fsdrCache/*"),
    "fsdr_controller": ("fsdrCtrl", "fsdrCtrl/*"),
    "control_interconnect": (
        "configRegs",
        "configRegs/*",
        "pipeline",
        "pipeline/*",
        "saesCtrl",
        "saesCtrl/*",
        "dramIF",
        "dramIF/*",
    ),
}

REQUIRED_REPORT_COMPONENTS = tuple(REPORT_COMPONENTS)

# The order is the paper table order.  Keep this as the single source of
# truth for the public proxy report rather than allowing tool-specific
# buckets to leak into the appendix.
TABLE4_COMPONENTS = (
    "mvu",
    "mvu_mmcu",
    "mvu_vector_alu",
    "mvu_bilinear_unit",
    "mvu_norm_unit",
    "mvu_activation_unit",
    "ggu_array",
    "ggu_position_calc",
    "ggu_cov_builder",
    "ggu_sh_op_generator",
    "fsdr_subsystem",
    "fsdr_lsh_hash_unit",
    "fsdr_cam_array",
    "fsdr_controller",
    "on_chip_buffers",
    "on_chip_buffers_weight_buffer",
    "on_chip_buffers_feature_buffer",
    "on_chip_buffers_tile_buffer",
    "control_and_clock",
    "control_interconnect",
    "pll_clock_tree",
    "io_phy",
    "io_lpddr4x_phy",
    "routing_filler",
    "total_die",
)

# Table 4 uses dynamic/static power for component rows, has area only for
# routing/filler, and prints a combined power number for the total-die row.
TABLE4_METRICS = {
    component: ("area_mm2", "dynamic_power_w", "static_power_w")
    for component in TABLE4_COMPONENTS
}
TABLE4_METRICS["routing_filler"] = ("area_mm2",)
TABLE4_METRICS["total_die"] = ("area_mm2", "total_power_w")

# This is the complete parent/child structure in the final paper table.  It
# deliberately includes groups that cannot be fully measured in the public
# ASAP7 flow (PLL and commercial I/O/PHY) so their row relationship never
# disappears from a public report.
PAPER_TABLE4_GROUPS = {
    "mvu": (
        "mvu_mmcu",
        "mvu_vector_alu",
        "mvu_bilinear_unit",
        "mvu_norm_unit",
        "mvu_activation_unit",
    ),
    "ggu_array": (
        "ggu_position_calc",
        "ggu_cov_builder",
        "ggu_sh_op_generator",
    ),
    "fsdr_subsystem": (
        "fsdr_lsh_hash_unit",
        "fsdr_cam_array",
        "fsdr_controller",
    ),
    "on_chip_buffers": (
        "on_chip_buffers_weight_buffer",
        "on_chip_buffers_feature_buffer",
        "on_chip_buffers_tile_buffer",
    ),
    "control_and_clock": (
        "control_interconnect",
        "pll_clock_tree",
    ),
    "io_phy": ("io_lpddr4x_phy",),
}

# Only these groups have complete, additive public ASAP7 measurements.  The
# parser must leave the Control & Clock and I/O & PHY parents unset instead of
# silently treating unavailable commercial collateral as zero.
TABLE4_AGGREGATES = {
    component: PAPER_TABLE4_GROUPS[component]
    for component in ("mvu", "ggu_array", "fsdr_subsystem", "on_chip_buffers")
}

TABLE4_LABELS = {
    "mvu": "MVU",
    "mvu_mmcu": "MMCU",
    "mvu_vector_alu": "VectorALU",
    "mvu_bilinear_unit": "BilinearUnit",
    "mvu_norm_unit": "NormUnit",
    "mvu_activation_unit": "ActivationUnit",
    "ggu_array": "GGU Array (x32 PEs)",
    "ggu_position_calc": "PositionCalc",
    "ggu_cov_builder": "CovBuilder",
    "ggu_sh_op_generator": "SH_OPGenerator",
    "fsdr_subsystem": "FSDR Subsystem",
    "fsdr_lsh_hash_unit": "LSHHashUnit",
    "fsdr_cam_array": "CAM Array (32-entry)",
    "fsdr_controller": "FSDR Controller",
    "on_chip_buffers": "On-chip Buffers",
    "on_chip_buffers_weight_buffer": "Weight Buffer (128 KB)",
    "on_chip_buffers_feature_buffer": "Feature Buffer (256 KB)",
    "on_chip_buffers_tile_buffer": "Tile Buffer (64 KB)",
    "control_and_clock": "Control & Clock",
    "control_interconnect": "Control + Interconnect",
    "pll_clock_tree": "PLL + Clock tree",
    "io_phy": "I/O & PHY",
    "io_lpddr4x_phy": "I/O + LPDDR4X PHY",
    "routing_filler": "Routing / filler",
    "total_die": "Total die",
}

BUFFER_PROXY_COMPONENTS = {
    "mem_1365x768": "on_chip_buffers_weight_buffer",
    "bank_65536x16": "on_chip_buffers_feature_buffer",
    "mem_16384x32": "on_chip_buffers_tile_buffer",
}

# ASAP7 has no commercial LPDDR4X PHY or PLL collateral in the artifact.
# These rows remain in the schema with null public values so the output has
# the same structure as the paper table without inventing measurements.
UNMODELED_COMPONENTS = {
    "control_and_clock",
    "pll_clock_tree",
    "io_phy",
    "io_lpddr4x_phy",
}
