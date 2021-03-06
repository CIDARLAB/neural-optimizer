from app.mod_dafd.helper_scripts.ModelHelper import ModelHelper
from app.mod_dafd.core_logic.RegimeClassifier import RegimeClassifier
from app.mod_dafd.core_logic.Regressor import Regressor


class ForwardModel:
	"""
	Bundles regime and regression models together.
	This is meant to be a kind of end interface.
	Plug in features, get predicted outputs. Simple!

	Works by constructing r*o forward models,
		where r is the number of regimes and o is the number of outputs (like droplet size and generation rate)

	When you want to predict outputs,
		1. Model predicts regime based on inputs
		2. For each output variable, the model returns a

	"""

	regime_classifier = None
	regressor = None

	def __init__(self,should_generate_regime_classifier=True):
		self.MH = ModelHelper.get_instance() # type: ModelHelper
		if should_generate_regime_classifier:
			self.regime_classifier = RegimeClassifier()
		self.model_dict = {}
		for regime in self.MH.regime_indices:
			for header in self.MH.output_headers:
				self.model_dict[header+str(regime)] = Regressor(header, regime)


	def predict(self, features, normalized = False, regime=0):
		# regime is an optional parameter that tells the prediction to override the regime prediction
		ret_dict = {}
		if normalized:
			norm_features = features
		else:
			norm_features = self.MH.normalize_set(features)
		if regime == 0:
			regime = self.regime_classifier.predict(norm_features)
		ret_dict["regime"] = regime
		for header in self.MH.output_headers:
			ret_dict[header] = self.model_dict[header+str(int(regime))].predict(norm_features)[0]
		return ret_dict


