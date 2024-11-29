// Packages

// Components
import { InferenceResults } from "./InferenceResults/InferenceResults";
import { InferenceTextBox } from "./InferenceTextBox/InferenceTextBox";

// Logic

// Context

// Services

// Styles
import "./InferencePage.css";

// Assets

export const InferencePage = () => {
	return (
		<div className='page inference-page'>
			<InferenceResults />
			<InferenceTextBox />
		</div>
	);
};
