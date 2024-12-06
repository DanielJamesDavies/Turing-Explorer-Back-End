// Packages
import { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";

// Components

// Logic

// Context
import { InferenceContext } from "../../../context/InferenceContext";
import { LatentContext } from "../../../context/LatentContext";
import { APIContext } from "../../../context/APIContext";

// Services

// Styles

// Assets

export const InferenceResultsLogic = () => {
	const { inferenceResults, viewingInferenceResultId, setViewingInferenceResultId, isViewingInferenceResult, setIsViewingInferenceResult } =
		useContext(InferenceContext);
	const { setLatentLayer, setLatentIndex } = useContext(LatentContext);
	const [sequenceOfThoughtsTokenIndex, setSequenceOfThoughtsTokenIndex] = useState(false);
	const [sequenceOfThoughtsTopLatents, setSequenceOfThoughtsTopLatents] = useState(false);
	const { APIRequest } = useContext(APIContext);
	const navigate = useNavigate();

	const onClickResultsItem = (inferenceResultId) => {
		setViewingInferenceResultId(inferenceResultId);
		setIsViewingInferenceResult(true);
	};

	const backToInferenceResults = () => {
		setIsViewingInferenceResult(false);
	};

	const goToLatent = (e, newLayer, newLatent) => {
		if (e?.button === 1)
			return window.open(window?.location?.origin + "/latent?layer=" + (newLayer + 1) + "&latent=" + (newLatent + 1), "_blank");
		setSearchBarValue("");
		setLatentLayer(newLayer);
		setLatentIndex(newLatent);
		navigate("/latent?layer=" + (newLayer + 1) + "&latent=" + (newLatent + 1));
	};

	const onClickToken = async (tokenIndex) => {
		setSequenceOfThoughtsTokenIndex(tokenIndex);
		setSequenceOfThoughtsTopLatents(false);

		const res = await APIRequest("/get-sequence-of-thoughts", "POST", {
			tokenIds: inferenceResults?.find((e) => e?.inference_id === viewingInferenceResultId)?.tokenIds,
			tokenIndex: tokenIndex,
		});
		if (!res?.success) {
			setSequenceOfThoughtsTokenIndex(false);
			setSequenceOfThoughtsTopLatents(false);
			return false;
		}

		let latent_relationships = res?.latent_relationships?.map((latent_relationship) => {
			return latent_relationship?.map((e, i) =>
				i === 0
					? e?.sort((a, b) => {
							return a[1] + a[0] * 40960 - (b[1] + b[0] * 40960);
					  })
					: e
			);
		});
		latent_relationships = latent_relationships?.map((e) => {
			return [
				e[0],
				latent_relationships?.filter((e2) => JSON.stringify(e2[0]) === JSON.stringify(e[0]))?.reduce((sum, e2) => sum + e2[1], 0),
			];
		});
		latent_relationships = latent_relationships?.filter(
			(e, i) => latent_relationships?.slice(0, i)?.findIndex((e2) => JSON.stringify(e2[0]) === JSON.stringify(e[0])) === -1
		);

		setSequenceOfThoughtsTopLatents(
			res?.top_latents?.map((layer_top_latents, layer_index) => {
				return layer_top_latents.map((top_latent) => {
					return {
						latent: top_latent?.latent,
						value: top_latent?.value,
						topSequences: top_latent?.topSequences,
						relationships: latent_relationships?.filter(
							(e) => e[0].findIndex((e2) => JSON.stringify(e2) === JSON.stringify([layer_index, top_latent?.latent])) !== -1
						),
					};
				});
			})
		);
	};

	const [latentsLatentWidth, setLatentsLatentWidth] = useState(1);

	const latentsLatentRef = (e) => {
		if (e) {
			setLatentsLatentWidth(e?.clientWidth);
		}
	};

	const [latentPositionMouseOver, setLatentPositionMouseOver] = useState(false);

	const onLatentMouseEnter = (layer_index, latent_index) => {
		setLatentPositionMouseOver([layer_index, latent_index]);
	};

	const onLatentMouseLeave = () => {
		setLatentPositionMouseOver(false);
	};

	return {
		inferenceResults,
		viewingInferenceResultId,
		isViewingInferenceResult,
		sequenceOfThoughtsTokenIndex,
		sequenceOfThoughtsTopLatents,
		onClickResultsItem,
		backToInferenceResults,
		goToLatent,
		onClickToken,
		latentsLatentRef,
		latentsLatentWidth,
		latentPositionMouseOver,
		onLatentMouseEnter,
		onLatentMouseLeave,
	};
};
