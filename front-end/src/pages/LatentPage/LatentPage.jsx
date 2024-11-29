// Packages

// Components
import { LatentDisplay } from "./LatentDisplay/LatentDisplay";
import { TopSequencesList } from "./TopSequencesList/TopSequencesList";
import { TopTokenFrequencies } from "./TopTokenFrequencies/TopTokenFrequencies";

// Logic

// Context

// Services

// Styles

// Assets

export const LatentPage = () => {
	return (
		<div className='page latent-page'>
			<LatentDisplay />
			<TopSequencesList />
			<TopTokenFrequencies />
		</div>
	);
};
