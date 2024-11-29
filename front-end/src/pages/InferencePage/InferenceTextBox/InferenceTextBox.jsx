// Packages

// Components
import { TextBox } from "../../../components/TextBox/TextBox";

// Logic
import { InferenceTextBoxLogic } from "./InferenceTextBoxLogic";

// Context

// Services

// Styles
import "./InferenceTextBox.css";

// Assets

export const InferenceTextBox = () => {
	const {
		inferenceTextBoxValue,
		setInferenceTextBoxValue,
		isGettingInferenceResults,
		inferenceResults,
		isViewingInferenceResult,
		submitInferenceRequest,
	} = InferenceTextBoxLogic();

	return (
		<div
			className={
				"inference-text-box-container" +
				(isGettingInferenceResults ? " inference-text-box-container-getting-results" : "") +
				(inferenceResults === false ? " inference-text-box-container-no-results" : "") +
				(isViewingInferenceResult ? " inference-text-box-container-viewing-result" : "")
			}
		>
			<div className='inference-text-box-title'>Turing-LLM Inference</div>
			<TextBox
				className='inference-text-box'
				value={inferenceTextBoxValue}
				setValue={setInferenceTextBoxValue}
				placeholder='Please type something for Turing-LLM to complete here'
				onEnter={submitInferenceRequest}
				focusOnStart={!isGettingInferenceResults}
				blurOnEnter={true}
			/>
			<div className='inference-text-box-loading-container'>
				<div className='inference-text-box-loading'>
					<div className='inference-text-box-loading-label'>Generating Completion...</div>
				</div>
			</div>
		</div>
	);
};
