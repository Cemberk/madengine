# Configuring distributed inference

Distributed inference workloads are configured four different ways. Three of them
already exist and are owned by the teams that authored them; madengine supports all
three unchanged. The fourth is madengine's own, for anyone who wants the whole
picture in one file.

The organising idea is **where in the pipeline a value is created**, because that is
also who reviews a change to it:

| way | file | created by | carries |
|-----|------|-----------|---------|
| 1 | `scripts/<dir>/models.yaml` | whoever tuned the model on that hardware | serve flags per model × role × mode, model env |
| 2 | `scripts/<dir>/configs/{default,perf,acc}.yaml` | whoever defines the measurement | benchmark kind, knobs, `extra_args`, env |
| 3 | `cluster.sh` | whoever provisioned the cluster | site facts as `${VAR:-default}` |
| 4 | `mad-config.yaml` | all three, in labelled sections | the same three layers, explicitly separated |

These are not four ways of doing one thing. Ways 1–3 are three *layers*, and they
overlap only where a team needed something another layer owned. Way 4 is those same
layers in one file.

Whatever you use, it ends up as **environment variables** reaching the workload,
because that is the only thing that survives both the container boundary and the
choice of launcher.

---

## Ways 1–3: what madengine does

**It routes, validates and documents. It does not parse them.**

`models.yaml`, `configs/*.yaml` and `cluster.sh` are read by the scripts that own
them, inside the container — madengine never interprets those schemas. That keeps
each format owned by its team, and means a team changing its own file cannot break
madengine.

What madengine does do is make sure the pointers reach the workload, and warn about
configurations that are genuinely ambiguous.

### The `args` trap

`args` on a model card means **two different things** depending on who runs it:

| consumer | treats `args` as |
|---|---|
| madengine (`deployment/slurm.py`) | arguments to the model script — `bash <script> <args>` |
| the STANDALONE Jenkins pipeline | arguments to `sbatch` |

So a distributed card with `args: "-N 2 -n 2"` hands those flags to the batch script
under madengine, and to `sbatch` under STANDALONE. madengine warns when it sees this:

```
⚠ pyt_large_ep_bench_2n: args='-N 2 -n 2' contains sbatch flag(s) ['-N', '-n'].
  madengine passes args to the model script, not to sbatch, so these reach the
  script as positional arguments. Node count for a distributed card comes from
  distributed.nnodes / slurm.nodes.
```

For a distributed card, put the node count in `distributed.nnodes` and `slurm.nodes`,
which both consumers read the same way. This is also why way 4 is pointed at by an
environment variable rather than by `args` — adding a third meaning to that field
would make the ambiguity worse.

---

## Way 4: `mad-config.yaml`

One file, three labelled sections, one per creation point.

```yaml
version: 1

site:                       # created by: whoever provisioned the cluster
  env:
    NVME_ROOT: /mnt/m2m_nobackup
    SHARED_MOUNT: /shared_inference

model:                      # created by: whoever tuned this model
  id: moonshotai/Kimi-K3    # how way 2 names it
  local_name: Kimi-K3       # how ways 1 and 3 name it (the on-disk directory)
  env:
    TP_SIZE: '8'
    PP_SIZE: '2'
  serve:
    base: "--attention-backend aiter"
    modes:
      tp: "--tensor-parallel-size 8"
    roles:
      prefill:
        tp: "--disable-cuda-graph"
      decode:
        tp: "--disable-radix-cache --cuda-graph-bs 8 16 32 64 128 256 512"

benchmark:                  # created by: whoever defines the measurement
  - env:                    # no 'kind' -> applies to every benchmark
      SEEDS: '3'
  - kind: niah
    env:
      NIAH_WORDS: '10000,50000,100000,200000'
```

### Where it is found

A sibling `mad-config.yaml` next to the card's script directory, or an explicit path
in `env_vars.MAD_CONFIG`. (The sibling-by-convention rule mirrors how the accuracy
work locates a sibling `acc.yaml` next to `--config`.)

No file means nothing changes — ways 1–3 are the common case and are untouched.

### Precedence

Lowest first. The last two are above the file so an operator pinning something at
submit time still wins, which is what all three existing formats already rely on:

```
site                                    from mad-config.yaml
model                                   from mad-config.yaml
benchmark  (task-level, then by kind)   from mad-config.yaml
model_info.env_vars                     the card's own env_vars
additional_context.env_vars             -e / submit-time override   (highest)
```

Layers **merge**, they do not replace: a key set only in `site` survives a `model`
section that does not mention it.

### Serve flags are a map, not a string

Within `serve`, a flag string is parsed into a map so a later layer can override a
*single* flag:

```yaml
base: "--attention-backend aiter --tp 1"    # -> {--attention-backend: aiter, --tp: '1'}
modes: {tp: "--tp 8"}                       # -> --tp becomes '8', aiter survives
```

This is the one thing way 4 does that way 1 cannot. `models.yaml` stores flags as an
opaque string, so overriding one flag means string surgery — its own comments note
that the moriio path "strips any duplicate from the yaml `tp:` string". A map removes
the need for that.

Flags taking several values are kept whole: `--cuda-graph-bs 8 16 32` stays
`'8 16 32'`, not three separate tokens.

### Why adopting it is safe

Way 4 resolves into **exactly the `env_vars` a card already carries**. It is a front
end, not a migration: no workload script changes, and no launcher needs to know the
file exists. `tests/unit/test_layered_config.py` asserts this directly — a way-4 file
and the equivalent hand-written `env_vars` block must resolve to identical
environments.

### Scope

Way 4 is resolved by madengine, so it is available on the madengine path. The
STANDALONE Jenkins pipeline deliberately runs without installing madengine, so cards
run that way should use ways 1–3.

---

## Choosing

- **Tuning a model's serve recipe** → way 1 (`models.yaml`); it owns the role × mode axis.
- **Defining what to measure** → way 2 (`configs/*.yaml`); it owns benchmark selection and inheritance.
- **Describing a cluster** → way 3 (`cluster.sh`); every value `${VAR:-default}` so the environment always wins.
- **Wanting all three in one reviewable file, on the madengine path** → way 4.

Related: [configuration.md](configuration.md), [launchers.md](launchers.md).
