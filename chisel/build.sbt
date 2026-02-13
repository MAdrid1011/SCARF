// SCARF Chisel RTL Build Configuration
ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "scarf"

lazy val root = (project in file("."))
  .settings(
    name := "scarf-rtl",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % "6.6.0",
      "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % "test",
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % "6.6.0" cross CrossVersion.full
    ),
  )
