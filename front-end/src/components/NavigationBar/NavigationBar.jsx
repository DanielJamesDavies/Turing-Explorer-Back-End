// Packages

// Components

// Logic
import { NavigationBarLogic } from "./NavigationBarLogic";

// Context

// Services

// Styles
import "./NavigationBar.css";

// Assets

export const NavigationBar = () => {
	const {
		goToHomePage,
		goToLatentPage,
		goToInferencePage,
		searchBarValue,
		changeSearchBarValue,
		onKeyDownSearchBar,
		decrementLayer,
		incrementLayer,
		decrementLatent,
		incrementLatent,
		latentLayerInputValue,
		latentIndexInputValue,
		changeLatentLayerInputValue,
		changeLatentIndexInputValue,
	} = NavigationBarLogic();

	return (
		<div className='navigation-bar'>
			{/* Title */}
			<div
				className={
					"navigation-bar-title" + (window?.location?.pathname.replace("/", "").trim() !== "" ? " navigation-bar-title-animation" : "")
				}
				onMouseDown={(e) => e?.preventDefault()}
				onClick={goToHomePage}
				onAuxClick={goToHomePage}
			>
				Turing-LLM Explorer
			</div>

			{window?.location?.pathname.replace("/", "").trim() === "" ? null : (
				<div className='navigation-bar-search-bar-container'>
					<input
						value={searchBarValue}
						onChange={changeSearchBarValue}
						onKeyDown={onKeyDownSearchBar}
						placeholder='Search Latents'
					></input>
				</div>
			)}

			<div className='navigation-bar-inference-btn-container'>
				<button
					className='navigation-bar-inference-btn'
					onMouseDown={(e) => e?.preventDefault()}
					onClick={goToInferencePage}
					onAuxClick={goToInferencePage}
				>
					<i className='fa-solid fa-message' />
					<span>Inference</span>
				</button>
			</div>

			<div className='navigation-bar-latent-navigation-container'>
				{/* Layer Navigation */}
				<div className='navigation-bar-latent-navigation-item navigation-bar-latent-navigation-item-layer'>
					<div className='navigation-bar-latent-navigation-item-icon'>
						<i className='fa-solid fa-layer-group' />
					</div>
					<button onClick={decrementLayer}>
						<i className='fa-solid fa-chevron-left' />
					</button>
					<div className='navigation-bar-latent-navigation-item-label'>Layer</div>
					<input value={latentLayerInputValue} onChange={changeLatentLayerInputValue} />
					<button onClick={incrementLayer}>
						<i className='fa-solid fa-chevron-right' />
					</button>
				</div>

				{/* Latent Navigation */}
				<div className='navigation-bar-latent-navigation-item navigation-bar-latent-navigation-item-latent'>
					<div className='navigation-bar-latent-navigation-item-icon'>
						<i className='fa-regular fa-circle' />
					</div>
					<button onClick={decrementLatent}>
						<i className='fa-solid fa-chevron-left' />
					</button>
					<div className='navigation-bar-latent-navigation-item-label'>Latent</div>
					<input value={latentIndexInputValue} onChange={changeLatentIndexInputValue} />
					<button onClick={incrementLatent}>
						<i className='fa-solid fa-chevron-right' />
					</button>
				</div>
			</div>
			{window?.location?.pathname === "/latent" ? null : (
				<div className='navigation-bar-latent-navigation-go-to-latent-btn-container'>
					<button
						className='navigation-bar-latent-navigation-go-to-latent-btn'
						onMouseDown={(e) => e?.preventDefault()}
						onClick={goToLatentPage}
						onAuxClick={goToLatentPage}
					>
						View Latent
					</button>
				</div>
			)}
		</div>
	);
};
