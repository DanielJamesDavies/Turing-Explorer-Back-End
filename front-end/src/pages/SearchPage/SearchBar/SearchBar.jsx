// Packages

// Components

// Logic
import { SearchBarLogic } from "./SearchBarLogic";

// Context

// Services

// Styles
import "./SearchBar.css";

// Assets

export const SearchBar = () => {
	const {
		searchBarValue,
		changeSearchBarValue,
		isGettingSearchResults,
		searchResults,
		searchBarRef,
		isSearchBarFocused,
		onClickSearchBar,
		onBlurSearchBar,
		onKeyDownSearchBar,
	} = SearchBarLogic();

	return (
		<div
			className={
				"search-bar-container" +
				(isGettingSearchResults ? " search-bar-container-searching" : "") +
				(searchResults === false ? " search-bar-container-not-searched" : "") +
				(searchBarValue?.length === 0 ? " search-bar-container-no-search" : "") +
				(isSearchBarFocused ? " search-bar-container-focused" : "")
			}
		>
			<div className='search-bar-title'>Explore Turing-LLM-1.0-254M</div>
			<div className='search-bar'>
				<input
					ref={searchBarRef}
					value={searchBarValue}
					onChange={changeSearchBarValue}
					onClick={onClickSearchBar}
					onBlur={onBlurSearchBar}
					onKeyDown={onKeyDownSearchBar}
				></input>
				<div className='search-bar-placeholder'>
					<i className='fa-solid fa-search' />
					<span>Search Latents</span>
				</div>
			</div>
			<div className='search-bar-searching-container'>
				<div className='search-bar-searching'>
					<div className='search-bar-searching-label'>Searching for Latents...</div>
				</div>
			</div>
		</div>
	);
};
