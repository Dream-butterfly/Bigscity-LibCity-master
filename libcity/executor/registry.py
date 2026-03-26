import importlib


EXECUTOR_REGISTRY = {
    "TrajLocPredExecutor": ("libcity.executor.traj_loc_pred_executor", "TrajLocPredExecutor"),
    "TrafficStateExecutor": ("libcity.executor.traffic_state_executor", "TrafficStateExecutor"),
    "DCRNNExecutor": ("libcity.executor.dcrnn_executor", "DCRNNExecutor"),
    "MTGNNExecutor": ("libcity.executor.mtgnn_executor", "MTGNNExecutor"),
    "HyperTuning": ("libcity.executor.hyper_tuning", "HyperTuning"),
    "GeoSANExecutor": ("libcity.executor.geosan_executor", "GeoSANExecutor"),
    "MapMatchingExecutor": ("libcity.executor.map_matching_executor", "MapMatchingExecutor"),
    "GEMLExecutor": ("libcity.executor.geml_executor", "GEMLExecutor"),
    "AbstractTraditionExecutor": ("libcity.executor.abstract_tradition_executor", "AbstractTraditionExecutor"),
    "ChebConvExecutor": ("libcity.executor.chebconv_executor", "ChebConvExecutor"),
    "LINEExecutor": ("libcity.executor.line_executor", "LINEExecutor"),
    "ETAExecutor": ("libcity.executor.eta_executor", "ETAExecutor"),
    "GensimExecutor": ("libcity.executor.gensim_executor", "GensimExecutor"),
    "SSTBANExecutor": ("libcity.executor.sstban_executor", "SSTBANExecutor"),
    "STTSNetExecutor": ("libcity.executor.sttsnet_executor", "STTSNetExecutor"),
    "FOGSExecutor": ("libcity.executor.fogs_executor", "FOGSExecutor"),
    "TESTAMExecutor": ("libcity.executor.testam_executor", "TESTAMExecutor"),
    "TimeMixerExecutor": ("libcity.executor.timemixer_executor", "TimeMixerExecutor"),
    "STSSLExecutor": ("libcity.executor.STSSL_executor", "STSSLExecutor"),
    "MegaCRNExecutor": ("libcity.executor.megacrn_executor", "MegaCRNExecutor"),
    "TrafformerExecutor": ("libcity.executor.trafformer_executor", "TrafformerExecutor"),
    "PDFormerExecutor": ("libcity.executor.pdformer_executor", "PDFormerExecutor"),
    "ASTGNNExecutor": ("libcity.executor.astgnn_executor", "ASTGNNExecutor"),
}


def get_executor_class(executor_name):
    executor_spec = EXECUTOR_REGISTRY.get(executor_name)
    if executor_spec is None:
        raise AttributeError("executor is not found")
    module_path, class_name = executor_spec
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
