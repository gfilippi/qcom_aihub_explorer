# qcom_aihub_explorer

Offline explorer of Qualcomm AI Hub model performance, with versioned snapshots
across AI Hub / QAIRT releases.

| Folder | Content |
|---|---|
| `src/` | the tool: data collection and page generation code — see [src/README.md](src/README.md) |
| `tools/` | self-contained helper scripts and upstream checkouts used by the tool (e.g. `fetch_aihub_repo.sh`) |
| `results/` | one self-contained snapshot per release, each with its own `index.html` |
| `docs/` | documentation (Markdown) |

Sources: [Qualcomm AI Hub models](https://aihub.qualcomm.com/models) ·
[qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)
