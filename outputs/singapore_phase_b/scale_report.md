# Singapore Phase B — 规模化验证报告

## Scale Validation（100 → 500 → 1000 agents，同一真实供给 + 冻结 S7-W3）

| 指标 | 100 | 500 | 1000 | 异常判定 |
|---|---|---|---|---|
| routing failures (planning fallbacks 合计) | 49 | 245 | 501 | — |
| failed trips (MATSim) | 0 | 0 | 0 | — |
| PT boardings | 12 | 55 | 127 | — |
| mean trip time (min) | 8.96 | 9.67 | 9.62 | — |
| road delay (s/passage) | 0.51 | 0.51 | 0.51 | — |
| congestion (慢行占比) | 0.0 | 0.0002 | 0.0 | — |
| runtime (s) | 29.4 | 26.5 | 27.6 | — |
| mode share (bike/car/pt/walk) | 0.18/0.33/0.06/0.43 | 0.18/0.29/0.06/0.48 | 0.19/0.29/0.06/0.47 | — |
| stuckAndAbort | 0 | 0 | 0 | — |

### 异常判据与结论

- 失败行程/滞留：全部规模为 0 —— 通过；
- mode share 跨规模最大漂移 < 5pp —— 通过；
- baseline 拥堵（慢行占比）均 < 20% —— 通过。


**总体判定：规模扩大无异常，可进入 Phase C。**

