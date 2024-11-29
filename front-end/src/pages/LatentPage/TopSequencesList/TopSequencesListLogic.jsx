// Packages
import { useContext, useEffect, useState } from "react";

// Components

// Logic

// Context
import { LatentContext } from "../../../context/LatentContext";

// Services

// Styles

// Assets

export const TopSequencesListLogic = () => {
	const { latentLayer, setLatentLayer, latentIndex, setLatentIndex, latentTopSequencesCount, topSequencesList, topOtherLatents } =
		useContext(LatentContext);
	const [isShowingAll, setIsShowingAll] = useState(false);
	const [sequenceIsShowingOtherLatents, setSequenceIsShowingOtherLatents] = useState(-1);
	const sequenceOtherLatentsTypes = [
		// { key: "latents_other_sae_latent_indices_avg_sequence_non_adj", name: "Average Sequence Non-Adjusted" },
		// { key: "latents_other_sae_latent_indices_top_token_non_adj", name: "Top Token Non-Adjusted" },
		{ key: "latents_other_sae_latent_indices_avg_sequence_adj", name: "Average Sequence Adjusted" },
		{ key: "latents_other_sae_latent_indices_top_token_adj", name: "Top Token Adjusted" },
	];
	const [sequenceOtherLatentsTypeIndex, setSequenceOtherLatentsTypeIndex] = useState(0);
	const [isHidingCommonOtherLatents, setIsHidingCommonOtherLatents] = useState(true);

	useEffect(() => {
		setIsShowingAll(false);
		setSequenceIsShowingOtherLatents(-1);
	}, [latentLayer, latentIndex]);

	const toggleIsShowingAll = () => {
		setIsShowingAll((oldValue) => !oldValue);
	};

	const goToLatent = (e, newLayerIndex, newLatentIndex) => {
		if (e?.button === 1) {
			return window.open(window.location?.origin + "/latent?layer=" + (newLayerIndex + 1) + "&latent=" + (newLatentIndex + 1), "_blank");
		}
		setLatentLayer(newLayerIndex);
		setLatentIndex(newLatentIndex);
	};

	const decrementOtherLatentsType = () => {
		setSequenceOtherLatentsTypeIndex((oldValue) => Math.max(0, oldValue - 1));
	};

	const incrementOtherLatentsType = () => {
		setSequenceOtherLatentsTypeIndex((oldValue) => Math.min(sequenceOtherLatentsTypes?.length - 1, oldValue + 1));
	};

	const toggleIsHidingCommonOtherLatents = () => {
		setIsHidingCommonOtherLatents((oldValue) => !oldValue);
	};

	return {
		latentTopSequencesCount,
		topSequencesList,
		topOtherLatents,
		isShowingAll,
		toggleIsShowingAll,
		goToLatent,
		sequenceIsShowingOtherLatents,
		setSequenceIsShowingOtherLatents,
		sequenceOtherLatentsTypes,
		sequenceOtherLatentsTypeIndex,
		decrementOtherLatentsType,
		incrementOtherLatentsType,
		isHidingCommonOtherLatents,
		toggleIsHidingCommonOtherLatents,
	};
};