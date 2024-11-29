// Packages
import { useContext } from "react";

// Components

// Logic

// Context
import { InferenceContext } from "../../../context/InferenceContext";

// Services

// Styles

// Assets

export const InferenceTextBoxLogic = () => {
	const {
		inferenceTextBoxValue,
		setInferenceTextBoxValue,
		isGettingInferenceResults,
		inferenceResults,
		isViewingInferenceResult,
		submitInferenceRequest,
	} = useContext(InferenceContext);

	return {
		inferenceTextBoxValue,
		setInferenceTextBoxValue,
		isGettingInferenceResults,
		inferenceResults,
		isViewingInferenceResult,
		submitInferenceRequest,
	};
};
