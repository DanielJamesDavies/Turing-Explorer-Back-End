import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { APIContext } from "./APIContext";

export const LatentContext = createContext();

const LatentProvider = ({ children }) => {
	const { APIRequest } = useContext(APIContext);
	const layerCount = 12;
	const latentCount = 40960;
	const latentFrequencyTokensCount = 512 * 64 * 11 * 256;
	const latentTopSequencesCount = 512 * 16 * 405;
	const [latentLayer, setLatentLayer] = useState(0);
	const [latentIndex, setLatentIndex] = useState(1);
	const [latentFrequency, setLatentFrequency] = useState("-");
	const [topSequencesList, setTopSequencesList] = useState([]);
	const [topOutputTokenFrequencies, setTopOutputTokenFrequencies] = useState([]);
	const [topLayerUnembedTokenFrequencies, setTopLayerUnembedTokenFrequencies] = useState([]);
	const [topOtherLatents, setTopOtherLatents] = useState([]);
	const location = useLocation();

	useEffect(() => {
		const params = new URL(window?.location?.href).searchParams;
		const newLatentLayer = (params?.get("layer") || 1) - 1;
		const newLatentIndex = (params?.get("latent") || 2) - 1;
		if (newLatentLayer !== 0) setLatentLayer(newLatentLayer);
		if (newLatentIndex !== 1) setLatentIndex(newLatentIndex);
	}, [setLatentLayer, setLatentIndex]);

	const latentPositionLastChangedTime = useRef(0);
	useEffect(() => {
		const thisLatentPositionLastChangedTime = Date.now();
		latentPositionLastChangedTime.current = JSON.parse(JSON.stringify(thisLatentPositionLastChangedTime));

		const getLatentData = async () => {
			setLatentFrequency("-");
			setTopSequencesList([]);
			setTopOutputTokenFrequencies([]);
			setTopLayerUnembedTokenFrequencies([]);
			setTopOtherLatents([]);

			if (location?.pathname !== "/latent") return false;

			const res = await APIRequest("/latent?layer=" + latentLayer + "&latent=" + latentIndex);

			if (JSON.stringify(latentPositionLastChangedTime.current) !== JSON.stringify(thisLatentPositionLastChangedTime)) return false;

			if (res?.latentFrequency !== undefined) setLatentFrequency(res?.latentFrequency);

			if (res?.topSequencesList !== undefined) setTopSequencesList(res?.topSequencesList);

			if (res?.postFromSequenceLatentData?.outputTokenFrequencies !== undefined)
				setTopOutputTokenFrequencies(res?.postFromSequenceLatentData?.outputTokenFrequencies);

			if (res?.postFromSequenceLatentData?.layerUnembedTokenFrequencies !== undefined)
				setTopLayerUnembedTokenFrequencies(res?.postFromSequenceLatentData?.layerUnembedTokenFrequencies);

			if (res?.postFromSequenceLatentData?.topOtherLatents !== undefined)
				setTopOtherLatents(res?.postFromSequenceLatentData?.topOtherLatents);
		};

		const timeout = setTimeout(() => getLatentData(), 200);

		return () => {
			clearTimeout(timeout);
		};
	}, [
		latentLayer,
		latentIndex,
		APIRequest,
		setLatentFrequency,
		setTopSequencesList,
		setTopOutputTokenFrequencies,
		setTopLayerUnembedTokenFrequencies,
		setTopOtherLatents,
		location,
	]);

	return (
		<LatentContext.Provider
			value={{
				layerCount,
				latentCount,
				latentFrequencyTokensCount,
				latentTopSequencesCount,
				latentLayer,
				setLatentLayer,
				latentIndex,
				setLatentIndex,
				latentFrequency,
				topSequencesList,
				topOutputTokenFrequencies,
				topLayerUnembedTokenFrequencies,
				topOtherLatents,
			}}
		>
			{children}
		</LatentContext.Provider>
	);
};

export default LatentProvider;
