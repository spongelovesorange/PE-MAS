# PE-MAS

PE-MAS is a power-electronics multi-agent design studio for isolated flyback converter workflows. It provides a browser Studio UI, FastAPI backend, requirement analysis, topology reasoning, magnetic and component advisory, PLECS integration, engineering evidence gates, and structured report generation.

PE-MAS is an engineering assistant. It keeps release decisions gated until the design has credible simulation, loop, thermal, EMI/safety, BOM source-quality, and human signoff evidence.

## Quick Start

```bash
conda activate pe-gpt
cp .env.example .env
make bootstrap
make run
```

Before running, edit `.env` and set real local values for:

```text
PE_MAS_HOST=<bind-host>
PE_MAS_PORT=<port>
```

For API-only mode:

```bash
make run-api
```

## Configuration

Local machine settings and credentials belong in `.env`, which is ignored by git.

```text
PE_MAS_APP_MODE=studio
PE_MAS_HOST=<bind-host>
PE_MAS_PORT=<port>
PE_MAS_RUNTIME_DIR=.pe_mas_runtime
PE_MAS_PLECS_BACKEND=auto
PLECS_RPC_URL=<plecs-rpc-url>
PE_MAS_LLM_CREDENTIAL=<provider-credential>
PE_MAS_LLM_ENDPOINT=<optional-provider-endpoint>
```

`PLECS_RPC_URL` is optional for planning and review flows, but required for PLECS-backed evidence.

## Repository Layout

```text
app/          FastAPI app factory, Studio runtime, API routers, schemas
core/         MAS workflow, agents, PE domain logic, knowledge services, PLECS adapters
data/         Checked-in topology and PLECS registry data
frontend/     Browser-based Studio UI
plecs-mcp/    Optional local PLECS MCP helper package
server.py     Product entrypoint
Makefile      Install, run, and clean commands
```

Local runtime artifacts are ignored and should not be committed:

```text
.env
.pe_mas_runtime/
__pycache__/
*.pyc
*.log
```

## Main Commands

```bash
make bootstrap  # install Python requirements
make run        # start Studio UI and MAS backend
make run-api    # start API-only backend
make clean      # remove local runtime and cache files
```

## Engineering Evidence

PE-MAS reports are designed around explicit evidence gates:

- PLECS validation matrix across line and load corners where models are available.
- Loop evidence for TL431/opto feedback, Bode response, PM/GM, and CTR aging corners.
- Thermal evidence for MOSFET, rectifier or SR, transformer, snubber, and capacitor ripple.
- EMI/safety evidence for input filter, layout, pre-scan fields, and safety-rated parts.
- BOM source-quality controls for custom magnetics and mains EMI/filter items.
- Human signoff before any release-ready claim.

Until these gates are closed, PE-MAS should present the design as an engineering review package rather than a production release.
