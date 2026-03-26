import importlib


def _load_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


TASK_MODEL_REGISTRY = {
    "traj_loc_pred": {
        "DeepMove": ("libcity.model.trajectory_loc_prediction.DeepMove", "DeepMove"),
        "RNN": ("libcity.model.trajectory_loc_prediction.RNN", "RNN"),
        "FPMC": ("libcity.model.trajectory_loc_prediction.FPMC", "FPMC"),
        "LSTPM": ("libcity.model.trajectory_loc_prediction.LSTPM", "LSTPM"),
        "STRNN": ("libcity.model.trajectory_loc_prediction.STRNN", "STRNN"),
        "TemplateTLP": ("libcity.model.trajectory_loc_prediction.TemplateTLP", "TemplateTLP"),
        "SERM": ("libcity.model.trajectory_loc_prediction.SERM", "SERM"),
        "ATSTLSTM": ("libcity.model.trajectory_loc_prediction.ATSTLSTM", "ATSTLSTM"),
        "STAN": ("libcity.model.trajectory_loc_prediction.STAN", "STAN"),
        "CARA": ("libcity.model.trajectory_loc_prediction.CARA", "CARA"),
        "GeoSAN": ("libcity.model.trajectory_loc_prediction.GeoSAN", "GeoSAN"),
        "HSTLSTM": ("libcity.model.trajectory_loc_prediction.HSTLSTM", "HSTLSTM"),
    },
    "traffic_state_pred": {
        "FreTS": ("libcity.model.traffic_flow_prediction.FreTS", "FreTS"),
        "AGCRN": ("libcity.model.traffic_flow_prediction.AGCRN", "AGCRN"),
        "ASTGCN": ("libcity.model.traffic_flow_prediction.ASTGCN", "ASTGCN"),
        "MSTGCN": ("libcity.model.traffic_flow_prediction.MSTGCN", "MSTGCN"),
        "ACFM": ("libcity.model.traffic_flow_prediction.ACFM", "ACFM"),
        "STResNet": ("libcity.model.traffic_flow_prediction.STResNet", "STResNet"),
        "STResNetCommon": ("libcity.model.traffic_flow_prediction.STResNetCommon", "STResNetCommon"),
        "ACFMCommon": ("libcity.model.traffic_flow_prediction.ACFMCommon", "ACFMCommon"),
        "ASTGCNCommon": ("libcity.model.traffic_flow_prediction.ASTGCNCommon", "ASTGCNCommon"),
        "MSTGCNCommon": ("libcity.model.traffic_flow_prediction.MSTGCNCommon", "MSTGCNCommon"),
        "ToGCN": ("libcity.model.traffic_flow_prediction.ToGCN", "ToGCN"),
        "CONVGCN": ("libcity.model.traffic_flow_prediction.CONVGCN", "CONVGCN"),
        "STDN": ("libcity.model.traffic_flow_prediction.STDN", "STDN"),
        "STSGCN": ("libcity.model.traffic_flow_prediction.STSGCN", "STSGCN"),
        "STNN": ("libcity.model.traffic_flow_prediction.STNN", "STNN"),
        "ResLSTM": ("libcity.model.traffic_flow_prediction.ResLSTM", "ResLSTM"),
        "DGCN": ("libcity.model.traffic_flow_prediction.DGCN", "DGCN"),
        "MultiSTGCnet": ("libcity.model.traffic_flow_prediction.MultiSTGCnet", "MultiSTGCnet"),
        "CRANN": ("libcity.model.traffic_flow_prediction.CRANN", "CRANN"),
        "CONVGCNCommon": ("libcity.model.traffic_flow_prediction.CONVGCNCommon", "CONVGCNCommon"),
        "DSAN": ("libcity.model.traffic_flow_prediction.DSAN", "DSAN"),
        "MultiSTGCnetCommon": ("libcity.model.traffic_flow_prediction.MultiSTGCnetCommon", "MultiSTGCnetCommon"),
        "STGODE": ("libcity.model.traffic_flow_prediction.STGODE", "STGODE"),
        "STNorm": ("libcity.model.traffic_flow_prediction.STNorm", "STNorm"),
        "ESG": ("libcity.model.traffic_flow_prediction.ESG", "ESG"),
        "SSTBAN": ("libcity.model.traffic_flow_prediction.SSTBAN", "SSTBAN"),
        "STTSNet": ("libcity.model.traffic_flow_prediction.STTSNet", "STTSNet"),
        "FOGS": ("libcity.model.traffic_flow_prediction.FOGS", "FOGS"),
        "RGSL": ("libcity.model.traffic_flow_prediction.RGSL", "RGSL"),
        "DSTAGNN": ("libcity.model.traffic_flow_prediction.DSTAGNN", "DSTAGNN"),
        "STPGCN": ("libcity.model.traffic_flow_prediction.STPGCN", "STPGCN"),
        "MultiSPANS": ("libcity.model.traffic_flow_prediction.MultiSPANS", "MultiSPANS"),
        "SimST": ("libcity.model.traffic_flow_prediction.SimST", "SimST"),
        "TimeMixer": ("libcity.model.traffic_flow_prediction.TimeMixer", "TimeMixer"),
        "STSSL": ("libcity.model.traffic_flow_prediction.STSSL", "STSSL"),
        "STWave": ("libcity.model.traffic_flow_prediction.STWave", "STWave"),
        "PDFormer": ("libcity.model.traffic_flow_prediction.PDFormer", "PDFormer"),
        "STGNCDE": ("libcity.model.traffic_flow_prediction.STGNCDE", "STGNCDE"),
        "ASTGNN": ("libcity.model.traffic_flow_prediction.ASTGNN", "ASTGNN"),
        "ASTGNNCommon": ("libcity.model.traffic_flow_prediction.ASTGNNCommon", "ASTGNNCommon"),
        "DCRNN": ("libcity.model.traffic_speed_prediction.DCRNN", "DCRNN"),
        "STGCN": ("libcity.model.traffic_speed_prediction.STGCN", "STGCN"),
        "GWNET": ("libcity.model.traffic_speed_prediction.GWNET", "GWNET"),
        "MTGNN": ("libcity.model.traffic_speed_prediction.MTGNN", "MTGNN"),
        "TGCLSTM": ("libcity.model.traffic_speed_prediction.TGCLSTM", "TGCLSTM"),
        "TGCN": ("libcity.model.traffic_speed_prediction.TGCN", "TGCN"),
        "TemplateTSP": ("libcity.model.traffic_speed_prediction.TemplateTSP", "TemplateTSP"),
        "RNN": ("libcity.model.traffic_speed_prediction.RNN", "RNN"),
        "Seq2Seq": ("libcity.model.traffic_speed_prediction.Seq2Seq", "Seq2Seq"),
        "AutoEncoder": ("libcity.model.traffic_speed_prediction.AutoEncoder", "AutoEncoder"),
        "ATDM": ("libcity.model.traffic_speed_prediction.ATDM", "ATDM"),
        "GMAN": ("libcity.model.traffic_speed_prediction.GMAN", "GMAN"),
        "GTS": ("libcity.model.traffic_speed_prediction.GTS", "GTS"),
        "HGCN": ("libcity.model.traffic_speed_prediction.HGCN", "HGCN"),
        "STAGGCN": ("libcity.model.traffic_speed_prediction.STAGGCN", "STAGGCN"),
        "STMGAT": ("libcity.model.traffic_speed_prediction.STMGAT", "STMGAT"),
        "DKFN": ("libcity.model.traffic_speed_prediction.DKFN", "DKFN"),
        "STTN": ("libcity.model.traffic_speed_prediction.STTN", "STTN"),
        "D2STGNN": ("libcity.model.traffic_speed_prediction.D2STGNN", "D2STGNN"),
        "FNN": ("libcity.model.traffic_speed_prediction.FNN", "FNN"),
        "STID": ("libcity.model.traffic_speed_prediction.STID", "STID"),
        "DMSTGCN": ("libcity.model.traffic_speed_prediction.DMSTGCN", "DMSTGCN"),
        "HIEST": ("libcity.model.traffic_speed_prediction.HIEST", "HIEST"),
        "STAEformer": ("libcity.model.traffic_speed_prediction.STAEformer", "STAEformer"),
        "TESTAM": ("libcity.model.traffic_speed_prediction.TESTAM", "TESTAM"),
        "MegaCRN": ("libcity.model.traffic_speed_prediction.MegaCRN", "MegaCRN"),
        "Trafformer": ("libcity.model.traffic_speed_prediction.Trafformer", "Trafformer"),
        "NEW_MODEL": ("libcity.model.new.new_model", "NEW_MODEL"),
        "CCRNN": ("libcity.model.traffic_demand_prediction.CCRNN", "CCRNN"),
        "DMVSTNet": ("libcity.model.traffic_demand_prediction.DMVSTNet", "DMVSTNet"),
        "STG2Seq": ("libcity.model.traffic_demand_prediction.STG2Seq", "STG2Seq"),
        "CSTN": ("libcity.model.traffic_od_prediction.CSTN", "CSTN"),
        "GEML": ("libcity.model.traffic_od_prediction.GEML", "GEML"),
        "GSNet": ("libcity.model.traffic_accident_prediction.GSNet", "GSNet"),
    },
    "map_matching": {
        "STMatching": ("libcity.model.map_matching.STMatching", "STMatching"),
        "IVMM": ("libcity.model.map_matching.IVMM", "IVMM"),
        "HMMM": ("libcity.model.map_matching.HMMM", "HMMM"),
    },
    "road_representation": {
        "ChebConv": ("libcity.model.road_representation.ChebConv", "ChebConv"),
        "LINE": ("libcity.model.road_representation.LINE", "LINE"),
        "GeomGCN": ("libcity.model.road_representation.GeomGCN", "GeomGCN"),
        "GAT": ("libcity.model.road_representation.GAT", "GAT"),
        "Node2Vec": ("libcity.model.road_representation.Node2Vec", "Node2Vec"),
        "DeepWalk": ("libcity.model.road_representation.DeepWalk", "DeepWalk"),
    },
    "eta": {
        "DeepTTE": ("libcity.model.eta.DeepTTE", "DeepTTE"),
        "TTPNet": ("libcity.model.eta.TTPNet", "TTPNet"),
    },
}


def get_model_class(task, model_name):
    task_registry = TASK_MODEL_REGISTRY.get(task)
    if task_registry is None:
        raise AttributeError("task is not found")
    model_spec = task_registry.get(model_name)
    if model_spec is None:
        raise AttributeError("model is not found")
    module_path, class_name = model_spec
    return _load_class(module_path, class_name)
