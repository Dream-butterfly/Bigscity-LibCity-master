import importlib


EVALUATOR_REGISTRY = {
    "TrajLocPredEvaluator": ("libcity.evaluator.traj_loc_pred_evaluator", "TrajLocPredEvaluator"),
    "TrafficStateEvaluator": ("libcity.evaluator.traffic_state_evaluator", "TrafficStateEvaluator"),
    "CARALocPredEvaluator": ("libcity.evaluator.cara_loc_pred_evaluator", "CARALocPredEvaluator"),
    "GeoSANEvaluator": ("libcity.evaluator.geosan_evaluator", "GeoSANEvaluator"),
    "MapMatchingEvaluator": ("libcity.evaluator.map_matching_evaluator", "MapMatchingEvaluator"),
    "RoadRepresentationEvaluator": ("libcity.evaluator.road_representation_evaluator", "RoadRepresentationEvaluator"),
    "ETAEvaluator": ("libcity.evaluator.eta_evaluator", "ETAEvaluator"),
    "TrafficAccidentEvaluator": ("libcity.evaluator.traffic_accident_evaluator", "TrafficAccidentEvaluator"),
}


def get_evaluator_class(evaluator_name):
    evaluator_spec = EVALUATOR_REGISTRY.get(evaluator_name)
    if evaluator_spec is None:
        raise AttributeError("evaluator is not found")
    module_path, class_name = evaluator_spec
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
