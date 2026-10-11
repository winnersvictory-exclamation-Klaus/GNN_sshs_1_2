# GNN 파동·진동 연구 시작 자료

이 폴더에는 요청한 선행연구 세 갈래, 총 다섯 편의 원문 PDF와 한글 설명, 그리고 송신–수신 파동 진단 가설을 검토하는 작은 재현 실험이 있다.

| 자료 | 내용 |
|---|---|
| `선행연구_읽기노트.md` | 각 논문의 방법·목표·우리 질문과의 관계 |
| `실험결과와_다음단계.md` | 더미 실험 설계, 숫자, 해석의 한계, 후속 연구 설계 |
| `experiment.py` | NumPy만 사용하는 결정적 실험 코드 |
| `pair_results.csv` | 송신–수신 노드쌍별 측정값 |
| `distance_summary.csv` | 최단거리별 Spearman 상관 |
| `papers/` | 아래 다섯 편의 공식 또는 저자 공개 원문 PDF |

원문 목록:

1. GraphCON (ICML 2022): `papers/graphcon_2022.pdf` — https://proceedings.mlr.press/v162/rusch22a.html
2. Wave-driven GNN with Energy Dynamics (IJCAI 2025): `papers/wave_driven_2025.pdf` — https://www.ijcai.org/proceedings/2025/732
3. Hearing the Clusters of a Graph (arXiv 2009; Automatica 2012): `papers/hearing_clusters_2009_2012.pdf` — https://arxiv.org/abs/0911.4729
4. A Dynamic Mode Decomposition Approach for Decentralized Spectral Clustering of Graphs (2022): `papers/wave_dmd_clustering_2022.pdf` — https://arxiv.org/abs/2203.00004
5. When Graph Neural Networks Meet Dynamic Mode Decomposition (ICLR 2025): `papers/dmd_gnn_2025.pdf` — https://arxiv.org/abs/2410.05593

실행은 이 폴더에서 `python3 experiment.py`이다. Python과 NumPy가 필요하다. 결과 CSV를 덮어쓴다. 코드의 분석 대상은 학습된 GNN이나 DMD가 아닌 선형 평균 집계와 이산 파동이므로, 결과의 의미는 `실험결과와_다음단계.md`의 한계를 함께 읽어야 한다.
