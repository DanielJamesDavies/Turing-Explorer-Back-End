// Packages
import { useContext } from "react";

// Components

// Logic

// Context
import { LatentContext } from "../../../context/LatentContext";

// Services

// Styles

// Assets

export const TopTokenFrequenciesLogic = () => {
	const { topOutputTokenFrequencies, topLayerUnembedTokenFrequencies } = useContext(LatentContext);

	return { topOutputTokenFrequencies, topLayerUnembedTokenFrequencies };
};
