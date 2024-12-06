// Packages
import { useContext, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

// Components

// Logic

// Context
import { SearchContext } from "../../context/SearchContext";
import { LatentContext } from "../../context/LatentContext";

// Services

// Styles

// Assets

export const NavigationBarLogic = () => {
	const { searchBarValue, setSearchBarValue, changeSearchBarValue, isGettingSearchResults, setSearchResults, getSearchResults } =
		useContext(SearchContext);
	const { layerCount, latentCount, latentLayer, setLatentLayer, latentIndex, setLatentIndex } = useContext(LatentContext);
	const navigate = useNavigate();

	const goToHomePage = (e) => {
		if (isGettingSearchResults) return false;
		if (e?.button === 1) return window.open(window.location.origin + "/", "_blank");
		setSearchBarValue("");
		setSearchResults(false);
		navigate("/");
	};

	const goToInferencePage = (e) => {
		if (e?.button === 1) return window.open(window.location.origin + "/inference", "_blank");
		setSearchBarValue("");
		navigate("/inference");
	};

	const onKeyDownSearchBar = (e) => {
		if (e?.key?.toLowerCase() === "enter") {
			goToHomePage();
			getSearchResults();
		}
	};

	const decrementLayer = () => {
		setLatentLayer((oldValue) => Math.max(0, oldValue - 1));
	};

	const incrementLayer = () => {
		setLatentLayer((oldValue) => Math.min(layerCount - 1, oldValue + 1));
	};

	const decrementLatent = () => {
		setLatentIndex((oldValue) => Math.max(0, oldValue - 1));
	};

	const incrementLatent = () => {
		setLatentIndex((oldValue) => Math.min(latentCount - 1, oldValue + 1));
	};

	const [latentLayerInputValue, setLatentLayerInputValue] = useState(latentLayer);
	const [latentIndexInputValue, setLatentIndexInputValue] = useState(latentIndex);

	useEffect(() => {
		setLatentLayerInputValue(latentLayer + 1);
	}, [latentLayer]);

	useEffect(() => {
		setLatentIndexInputValue(latentIndex + 1);
	}, [latentIndex]);

	const changeLatentLayerInputValueTimeout = useRef(false);
	const changeLatentLayerInputValue = (e) => {
		setLatentLayerInputValue(e.target.value);

		clearTimeout(changeLatentLayerInputValueTimeout.current);
		changeLatentLayerInputValueTimeout.current = setTimeout(() => {
			if (!isNaN(parseInt(e.target.value))) {
				setLatentLayer(Math.min(layerCount - 1, Math.max(0, parseInt(e.target.value) - 1)));
			}
		}, 1000);
	};

	const changeLatentIndexInputValueTimeout = useRef(false);
	const changeLatentIndexInputValue = (e) => {
		setLatentIndexInputValue(e.target.value);

		clearTimeout(changeLatentIndexInputValueTimeout.current);
		changeLatentIndexInputValueTimeout.current = setTimeout(() => {
			if (!isNaN(parseInt(e.target.value))) {
				setLatentIndex(Math.min(latentCount - 1, Math.max(0, parseInt(e.target.value) - 1)));
			}
		}, 1000);
	};

	const goToLatentPage = (e) => {
		if (e?.button === 1) {
			let newLatentLayer = latentLayer;
			let newLatentIndex = latentIndex;
			if (!isNaN(parseInt(latentLayerInputValue))) newLatentLayer = parseInt(latentLayerInputValue) - 1;
			if (!isNaN(parseInt(latentIndexInputValue))) newLatentIndex = parseInt(latentIndexInputValue) - 1;
			return window.open(window.location.origin + "/latent?layer=" + (newLatentLayer + 1) + "&latent=" + (newLatentIndex + 1), "_blank");
		}
		setSearchBarValue("");
		navigate("/latent");
	};

	return {
		goToHomePage,
		goToLatentPage,
		goToInferencePage,
		searchBarValue,
		changeSearchBarValue,
		onKeyDownSearchBar,
		layerCount,
		latentCount,
		latentLayer,
		decrementLayer,
		incrementLayer,
		latentIndex,
		decrementLatent,
		incrementLatent,
		latentLayerInputValue,
		latentIndexInputValue,
		changeLatentLayerInputValue,
		changeLatentIndexInputValue,
	};
};
