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

	const goToHomePage = () => {
		if (isGettingSearchResults) return false;
		setSearchBarValue("");
		setSearchResults(false);
		navigate("/");
	};

	const goToLatentPage = () => {
		setSearchBarValue("");
		navigate("/latent");
	};

	const goToInferencePage = () => {
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
