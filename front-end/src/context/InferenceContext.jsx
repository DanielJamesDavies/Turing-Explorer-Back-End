import React, { createContext, useCallback, useState, useEffect, useContext } from "react";
import { useLocation } from "react-router-dom";
import { APIContext } from "./APIContext";

export const InferenceContext = createContext();

const InferenceProvider = ({ children }) => {
	const { APIRequest } = useContext(APIContext);
	const [inferenceTextBoxValue, setInferenceTextBoxValue] = useState("");
	const [isGettingInferenceResults, setIsGettingInferenceResults] = useState(false);
	const [inferenceResults, setInferenceResults] = useState(false);
	const [viewingInferenceResultIndex, setViewingInferenceResultIndex] = useState(false);
	const [isViewingInferenceResult, setIsViewingInferenceResult] = useState(false);
	const location = useLocation();

	const submitInferenceRequest = useCallback(async () => {
		setIsGettingInferenceResults(true);
		const newInferenceTextBoxValue = JSON.parse(JSON.stringify(inferenceTextBoxValue));
		const res = await APIRequest("/inference", "POST", { prompt: newInferenceTextBoxValue });
		setIsGettingInferenceResults(false);
		setInferenceTextBoxValue("");
		setInferenceResults((oldValue) => (oldValue || []).concat([{ tokenIds: res?.response_tokens, tokens: res?.response_tokens_decoded }]));
	}, [inferenceTextBoxValue]);

	useEffect(() => {
		if (location?.pathname !== "/inference") {
			setIsViewingInferenceResult(false);
		}
	}, [location]);

	return (
		<InferenceContext.Provider
			value={{
				inferenceTextBoxValue,
				setInferenceTextBoxValue,
				isGettingInferenceResults,
				setIsGettingInferenceResults,
				inferenceResults,
				setInferenceResults,
				submitInferenceRequest,
				viewingInferenceResultIndex,
				setViewingInferenceResultIndex,
				isViewingInferenceResult,
				setIsViewingInferenceResult,
			}}
		>
			{children}
		</InferenceContext.Provider>
	);
};

export default InferenceProvider;
