# Third-party corpus attribution and notices

These independent corpus packages are provided for non-commercial course research.
No common license is applied to all datasets. Each upstream license and notice continues to apply.
Upstream README files are included byte-for-byte in each raw archive, including supplied citations,
license declarations and disclaimers. The exact upstream commits and per-file hashes are in manifest.json.

## ToxiCN and its lexicon

Creators: Junyu Lu, Bo Xu, Xiaokun Zhang, Changrong Min, Liang Yang, Hongfei Lin.
Paper: *Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmarks* (ACL 2023).
Source: https://github.com/DUT-lujunyu/ToxiCN
Paper: https://aclanthology.org/2023.acl-long.898/
License declared by upstream: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
License: https://creativecommons.org/licenses/by-nc-nd/4.0/
Legal code: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en
Upstream additionally states that resources are solely for scientific research and prohibits commercial use.

Published content: byte-identical upstream data, lexicons and README inside toxicn-raw.zip.
The ZIP wrapper is added for transport; no source text or labels are edited.
Derived classification, mixed datasets and annotation queues are generated locally and are not redistributed here.
The token ID arrays in the lexicon are not human definitions or sentiment labels.

## COLD

Creators: Jiawen Deng, Jingyan Zhou, Hao Sun, Fei Mi, Minlie Huang.
Paper: *COLD: A Benchmark for Chinese Offensive Language Detection* (EMNLP 2022).
Source: https://github.com/thu-coai/COLDataset
Paper: https://aclanthology.org/2022.emnlp-main.796/
License: Apache License 2.0; the upstream full LICENSE is included in cold-raw.zip.
The supplied copyright, attribution and citation notices are preserved in its README and LICENSE.

The separate classified package adds row IDs, provenance, categories and original/variant group metadata,
and normalizes file structure. Source text and labels are preserved; no sentiment labels are inferred.

## HED-COLD

Creators listed in upstream citation: Junqi Wu, Shujie Ji, Kang Zhong, Huiling Peng,
Zhendongxiao, Xiongding Liu, Wu Wei.
Paper: *Enhancing Chinese Offensive Language Detection with Homophonic Perturbation* (EMNLP 2025).
Source: https://github.com/sjie320/HED_COLDataset
Paper: https://aclanthology.org/2025.emnlp-main.1154/
Upstream README declares MIT; that declaration and author citation are retained verbatim in hed_cold-raw.zip.
This resource is derived from COLD. Its underlying COLD attribution and Apache notices are also retained
in cold-raw.zip and the COLD section of this document; the underlying COLD material is not relicensed.

The separate classified package adds metadata and verified links to COLD originals. It preserves the
upstream variant text, original text and offensive labels and does not claim to be an official new benchmark.

## OCNLI

Creators: Hai Hu, Kyle Richardson, Liang Xu, Lu Li, Sandra Kuebler, Larry Moss.
Paper: *OCNLI: Original Chinese Natural Language Inference* (Findings of EMNLP 2020).
Source: https://github.com/CLUEbenchmark/OCNLI
Paper: https://arxiv.org/abs/2010.05444
License declared by upstream: Creative Commons Attribution-NonCommercial 2.0 Generic.
License: https://creativecommons.org/licenses/by-nc/2.0/
The upstream README also identifies its LCMC news source and associated permission; that notice is preserved.

The separate classified package adds provenance and regex-based negation/contrast candidate tags,
preserves sentence pairs and NLI labels, and distinguishes unlabelled test pairs and no-consensus labels.
Subset training files are kept only in the raw archive; the largest training set alone is normalized.
The labels and candidate tags are not sentiment annotations or verified semantic negation annotations.
