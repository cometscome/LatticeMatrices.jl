# D†D CG benchmark

`ddagd_cg.jl` は

```text
(D†D) x = φ
```

を共役勾配法（CG）で解き、計時後に `D†D*x` を独立に再計算して
`||D†D*x-φ||/||φ||` を検証する、単一のJuliaスクリプトです。

選択できる演算子は staggered、Wilson、Wilson clover、HISQ、Shamir
domain wall、Möbius domain wall、generalized domain wall です。domain-wall
系の `D` は生の5次元 `D5DW` カーネルです。高水準の有効演算子
`D(m)D(1)⁻¹` に含まれるPauli–Villars内部反復は計測しません。

## バックエンドと初期セットアップ

ベンチマークでローカル開発版が混入しないよう、すべてJulia General
registryの公式リリース版を使用します。パッケージのバージョン番号はコード
内で固定しません。`--setup` の実行時点で、使用中のJuliaと相互に互換性が
あるLatticeMatrices、Gaugefields、LatticeDiracOperators、JACC、MPIの最新版を
解決します。同じバックエンドで `--setup` を再実行すると `Pkg.update()` も
行い、その時点の最新互換版へ環境を更新します。

次の5種類を同じスクリプトの `--backend` で選べます。

| `--backend` | 対象 |
|---|---|
| `threads` | マルチスレッドCPU |
| `cuda` | NVIDIA GPU / CUDA.jl |
| `amdgpu` | AMD GPU / AMDGPU.jl (ROCm) |
| `oneapi` | Intel GPU / oneAPI.jl |
| `metal` | Apple GPU / Metal.jl |

使用するマシンで、必要なバックエンドだけをまずセットアップします。同じ
コマンドを後日再実行すると、その時点の最新版へ更新できます。

```bash
julia ddagd_cg.jl --setup --backend=threads
julia ddagd_cg.jl --setup --backend=cuda
julia ddagd_cg.jl --setup --backend=amdgpu
julia ddagd_cg.jl --setup --backend=oneapi
julia ddagd_cg.jl --setup --backend=metal
```

バックエンドごとの環境は
`.environments/general-latest-compatible/<backend>` に生成され、Gitには
含まれません。ローカルの `LatticeMatrices-v1.1.0`、
`Gaugefields-v1`、`LatticeDiracOperators-v1` は参照しません。これにより、
開発中のworking treeや未登録変更がベンチマークへ混入しません。

別マシンへは `ddagd_cg.jl` だけをコピーしても実行できます。移動先で使う
バックエンドについて `--setup` を実行すると、その時点の最新互換版が
registryから導入されます。インターネットへ接続できない計算ノードでは、
事前にlogin nodeでセットアップするか、Julia depot/artifact cacheも転送
してください。

実際に解決された主要パッケージとGPUパッケージのバージョンは、セットアップ
完了時とベンチマーク開始時に表示され、CSVにも保存されます。後から完全に同じ
依存関係を再現する必要がある場合は、結果CSVと一緒に該当バックエンドの
`Project.toml` と `Manifest.toml` を保存してください。保存済みManifestを再現に
使う場合は `Pkg.instantiate()` のみを実行し、更新を行う `--setup` は実行しません。

## GPU選択

既定の `--devices=auto` を推奨します。スケジューラ、コンテナ、または
ユーザーが既に設定した可視デバイスを変更せず、各ノード内MPI rankを
可視GPUへ順番に1対1で割り当てます。各rankに異なるGPUが1枚だけ公開
されている構成もそのまま扱えます。

現在の可視デバイスと、このスクリプトが選んだデバイスは次で確認できます。

```bash
julia ddagd_cg.jl --backend=cuda   --list-devices
julia ddagd_cg.jl --backend=amdgpu --list-devices
julia ddagd_cg.jl --backend=oneapi --list-devices
julia ddagd_cg.jl --backend=metal  --list-devices
```

`--devices=2,3` のような明示指定も可能です。番号は必ず
`--list-devices` が表示する「そのプロセスから可視なデバイス」の0始まり
ordinalです。マシン名やGPUモデルとordinalの対応をコードには固定して
いません。NVIDIAの `CUDA_VISIBLE_DEVICES` やAMDの
`HIP_VISIBLE_DEVICES` などで外側から可視範囲を絞った場合、その範囲内で
再び0から数えます。

Metalは現状、1ノードにつき1 MPI rankのみ対応し、既定精度はsingleです。
Apple GPUはFloat64カーネルを扱えないため `--precision=double` は実行前に
エラーにします。Intel GPUは機種によってFloat64対応が異なるため、未知の
機種ではまず `--precision=single` を使用してください。

## 実行例

CPU、16スレッド、Wilson clover（`--threads=16` が必要なら自動で再起動）:

```bash
julia -t 16 ddagd_cg.jl \
  --backend=threads --threads=16 \
  --operator=wilson-clover --csw=1.0 \
  --lattice=16,16,16,32 --repeats=3 \
  --output=results.csv
```

2 GPU、1 GPU/MPI rank、HISQ（GPUベンダーを問わない形）:

```bash
julia -t 4 ddagd_cg.jl \
  --ranks=2 --backend=amdgpu --threads=4 \
  --operator=hisq --lattice=32,32,32,64 \
  --grid=2,1,1,1 --repeats=3 \
  --output=results.csv
```

`--backend=amdgpu` を `cuda` または `oneapi` に変えれば同じ計算条件で
比較できます。`--ranks=N` はMPI.jlとABIが一致するlauncherを使用します。
クラスタの互換 `mpiexec` やschedulerから直接起動する場合は `--ranks` を
省略してください。

MPI process gridは積がrank数に等しく、各格子長が対応する分割数で割り
切れれば変更できます。例えば4 rankなら、格子サイズに応じて
`--grid=4,1,1,1` と `--grid=2,2,1,1` の両方を選べます。

Möbius domain wall:

```bash
julia ddagd_cg.jl --ranks=2 --backend=cuda \
  --operator=mobius-domain-wall \
  --lattice=16,16,16,32 --grid=2,1,1,1 \
  --l5=12 --mass=0.1 --domain-wall-height=-1 \
  --mobius-b=2 --mobius-c=1
```

generalized domain wallの係数は1個（第5方向へbroadcast）またはL5個を
指定できます。

```bash
julia ddagd_cg.jl --operator=general-domain-wall \
  --lattice=8,8,8,16 --l5=4 \
  --a5=1 --b5=1.4,1.5,1.6,1.7 --c5=0.4,0.5,0.6,0.7
```

Juliaのthread数は起動時に決まるため、通常起動で `--threads=N` が現在値と
異なる場合、スクリプト自身を適切なthread数で一度だけ再起動します。
`julia -t N` を明示しても構いません。外部の `mpiexec` / schedulerから起動
する場合だけは、各rankのJulia起動コマンドに `-t N` も指定してください。

## ゲージ場とφ

既定は `--gauge=hot --seed=1234` です。各格子点・各方向について3×3複素
行列の実部と虚部を独立な一様分布 `[-0.5, 0.5)` で埋め、行を直交規格化し、
第3行を複素外積で構成してSU(3)へ再ユニタリ化します。乱数はPhilox4x32の
site-local streamで、同じseedならMPI分割を変えても同じ大域ゲージ場です。
これはCGベンチマーク用のhot startであり、heatbath/HMCで熱化された物理的
配位でもHaar分布から直接生成した配位でもありません。

`--gauge=cold` では全リンク `Uμ(x)` が3×3単位行列です。ゲージ場の境界は
周期的で、fermionの境界条件は空間3方向が周期的、時間方向が反周期的
`(1,1,1,-1)` です。

右辺 `φ` は別stream（`seed+1`）の複素Gaussian乱数で、実部・虚部はそれぞれ
標準偏差 `sqrt(1/2)`、したがって各複素成分について `E[|φ|²]=1` です。

## 判定と出力

`VERIFY` 行の `passed=true` は、CGが収束し、独立に再計算した相対残差が
`--verify-rtol` 以下だったことを表します。既定値は `10*rtol` です。
`RESULT` はbatch log向けの短い結果、`--output` はrank 0からCSVへ全metadata
と時間を追記します。計時範囲はCG本体とMPI/GPU同期を含み、ゲージ場構築、
JIT warm-up、最後の検算multiplyを含みません。

全オプションは次で確認できます。

```bash
julia ddagd_cg.jl --help
```
