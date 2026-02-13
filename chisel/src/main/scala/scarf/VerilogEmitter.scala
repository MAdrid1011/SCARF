package scarf

import chisel3._
import circt.stage.ChiselStage

/**
 * VerilogEmitter — Generate per-module Verilog files.
 *
 * Generates SystemVerilog for each SCARF module into the `generated/` directory.
 * Each module produces a separate .sv file for synthesis tool compatibility.
 *
 * Usage: sbt "runMain scarf.VerilogEmitter"
 */
object VerilogEmitter extends App {
  val outputDir = "generated"

  println(s"Generating SCARF Verilog to $outputDir/...")

  // Generate top-level (includes all sub-modules)
  // --split-verilog requires -o=<dir> for firtool output directory
  ChiselStage.emitSystemVerilogFile(
    new ScarfTop,
    Array("--target-dir", outputDir),
    Array(
      "--split-verilog",
      s"-o=$outputDir",
      "--lowering-options=disallowLocalVariables",
    ),
  )

  println(s"Done. Verilog files written to $outputDir/")
}
