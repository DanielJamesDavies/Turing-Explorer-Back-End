// Packages
import { useContext } from "react";

// Components

// Logic

// Context
import { LatentContext } from "../../../context/LatentContext";

// Services

// Styles

// Assets

export const LatentDisplayLogic = () => {
	const { latentFrequencyTokensCount, latentLayer, latentIndex, latentFrequency } = useContext(LatentContext);

	return { latentFrequencyTokensCount, latentLayer, latentIndex, latentFrequency };
};
