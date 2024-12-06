import React, { createContext, useCallback, useState, useEffect, useContext } from "react";
import { useLocation } from "react-router-dom";
import { APIContext } from "./APIContext";

export const InferenceContext = createContext();

const InferenceProvider = ({ children }) => {
	const { APIRequest } = useContext(APIContext);
	const [inferenceTextBoxValue, setInferenceTextBoxValue] = useState("");
	const [isGettingInferenceResults, setIsGettingInferenceResults] = useState(false);
	const [inferenceResults, setInferenceResults] = useState(false);
	const [viewingInferenceResultId, setViewingInferenceResultId] = useState(false);
	const [isViewingInferenceResult, setIsViewingInferenceResult] = useState(false);
	const location = useLocation();

	const submitInferenceRequest = useCallback(async () => {
		setIsGettingInferenceResults(true);
		const newInferenceTextBoxValue = JSON.parse(JSON.stringify(inferenceTextBoxValue));
		const handleStreamResponse = (res) => {
			console.log(res);
			if (res?.first) {
				setIsGettingInferenceResults(false);
				setInferenceTextBoxValue("");
				setViewingInferenceResultId(res?.inference_id);
				setIsViewingInferenceResult(true);
			}
			setInferenceResults((oldValue) =>
				(oldValue || [])
					?.filter((e) => e?.inference_id !== res?.inference_id)
					.concat([{ inference_id: res?.inference_id, tokenIds: res?.response_tokens, tokens: res?.response_tokens_decoded }])
			);
		};
		APIRequest("/inference", "POST", { prompt: newInferenceTextBoxValue }, handleStreamResponse);
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
				viewingInferenceResultId,
				setViewingInferenceResultId,
				isViewingInferenceResult,
				setIsViewingInferenceResult,
			}}
		>
			{children}
		</InferenceContext.Provider>
	);
};

export default InferenceProvider;
