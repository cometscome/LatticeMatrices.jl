import qex
import gauge
import gauge/stoutsmear
import os
import strformat

const lattice = [4, 4, 4, 4]

proc writeStage(output: File; stage: string; fields: auto; lo: auto) =
  let nc = fields[0]{0}.nrows
  for x3 in 0..<lattice[3]:
    for x2 in 0..<lattice[2]:
      for x1 in 0..<lattice[1]:
        for x0 in 0..<lattice[0]:
          let
            coords = [x0, x1, x2, x3]
            site = lo.rankIndex(coords).index
          for mu in 0..<4:
            for row in 0..<nc:
              for col in 0..<nc:
                let z = fields[mu]{site}[row, col]
                output.writeLine(
                  &"{stage}\t{mu + 1}\t{x0}\t{x1}\t{x2}\t{x3}\t" &
                  &"{row + 1}\t{col + 1}\t{z.re[][]:.17e}\t{z.im[][]:.17e}")

proc main() =
  qexInit()
  if paramCount() != 1:
    echo "usage: qex_stout_reference OUTPUT.tsv"
    qexFinalize()
    quit(2)

  let lo = newLayout(lattice)
  var
    thin = lo.newGauge()
    smeared = lo.newGauge()
    left = lo.newGauge()
    force = lo.newGauge()
    rng = lo.newRNGField(RngMilc6, 424242'u64)
    smearing = lo.newStoutSmear(0.1)

  threads:
    thin.random rng
    left.randomTAH rng
  smearing.smear(thin, smeared)
  smearing.smearDeriv(force, left)

  var output = open(paramStr(1), fmWrite)
  output.writeLine("stage\tmu\tx0\tx1\tx2\tx3\trow\tcol\tre\tim")
  writeStage(output, "thin", thin, lo)
  writeStage(output, "smeared", smeared, lo)
  writeStage(output, "left", left, lo)
  writeStage(output, "force", force, lo)
  output.close()
  qexFinalize()

main()
