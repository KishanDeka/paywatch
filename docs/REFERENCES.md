# Primary references

- [ULB/Kaggle credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): dataset scope, anonymized PCA columns, time, amount, labels.
- [PyTorch LSTMCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html): gate equations and reference state semantics. PayWatch implements those equations directly.
- [aiokafka consumer documentation](https://aiokafka.readthedocs.io/en/stable/consumer.html): manual commits and rebalance considerations.
- [aiokafka manual-commit example](https://aiokafka.readthedocs.io/en/stable/examples/manual_commit.html): commit only after processing and limits of delivery guarantees.
- [Apache Kafka releases](https://kafka.apache.org/community/downloads/): official Docker image versions.
- [Apache Kafka 4.0 upgrade documentation](https://kafka.apache.org/41/getting-started/upgrade/): removal of ZooKeeper in 4.x; this repository uses a pinned 3.9.2 KRaft configuration.
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html): inference sessions and execution providers.
- [ONNXMLTools source](https://github.com/onnx/onnxmltools): XGBoost conversion.
- [SQLAlchemy async documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html): asynchronous engines and connections.

Checked during repository creation, 2026-09-06. Pinned package versions are a locally tested
compatibility baseline, not a latest-version claim. Upgrade in a separate change and rerun export
parity, behavioral tests, and integration tests.
