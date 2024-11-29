// Packages
import { useContext, useLayoutEffect, useRef, useState } from "react";

// Components

// Logic

// Context
import { SearchContext } from "../../../context/SearchContext";

// Services

// Styles

// Assets

export const SearchBarLogic = () => {
	const {
		searchBarValue,
		changeSearchBarValue,
		searchResults,
		isGettingSearchResults,
		getSearchResults,
		searchWeights,
		setSearchWeights,
		isShowingSearchWeightSliders,
		setIsShowingSearchWeightSliders,
	} = useContext(SearchContext);
	const searchBarRef = useRef();
	const [isSearchBarFocused, setIsSearchBarFocused] = useState(true);

	useLayoutEffect(() => {
		searchBarRef?.current?.focus();
	}, [searchBarRef]);

	const onClickSearchBar = () => {
		setIsSearchBarFocused(true);
	};

	const onBlurSearchBar = () => {
		setIsSearchBarFocused(false);
	};

	const onKeyDownSearchBar = (e) => {
		if (e?.key?.toLowerCase() === "enter") {
			searchBarRef?.current?.blur();
			getSearchResults();
		}
	};

	const onChangeSearchWeight = (e, weightId) => {
		setSearchWeights((oldSearchWeights) => {
			let newSearchWeights = JSON.parse(JSON.stringify(oldSearchWeights));
			const index = newSearchWeights?.findIndex((e) => e?.id === weightId);
			if (index === -1) return newSearchWeights;
			newSearchWeights[index].value = e?.target?.value;
			return newSearchWeights;
		});
	};

	const toggleIsShowingSearchWeightSliders = () => {
		setIsShowingSearchWeightSliders((oldValue) => !oldValue);
	};

	return {
		searchBarValue,
		changeSearchBarValue,
		isGettingSearchResults,
		searchResults,
		searchBarRef,
		isSearchBarFocused,
		searchWeights,
		onChangeSearchWeight,
		onClickSearchBar,
		onBlurSearchBar,
		onKeyDownSearchBar,
		isShowingSearchWeightSliders,
		toggleIsShowingSearchWeightSliders,
	};
};
