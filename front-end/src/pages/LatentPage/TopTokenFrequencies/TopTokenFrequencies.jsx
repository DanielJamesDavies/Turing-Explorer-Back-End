// Packages

// Components

// Logic
import { TopTokenFrequenciesLogic } from "./TopTokenFrequenciesLogic";

// Context

// Services

// Styles
import "./TopTokenFrequencies.css";
import { LoadingCircle } from "../../../components/LoadingCircle/LoadingCircle";

// Assets

export const TopTokenFrequencies = () => {
	const { topOutputTokenFrequencies, topLayerUnembedTokenFrequencies } = TopTokenFrequenciesLogic();

	return (
		<div className='latent-top-token-frequencies-container'>
			<div className='subtitle'>List of Top Tokens on Latent Activated Sequences</div>
			<div className='light-text'>
				The following is a list of top token-frequency pairs. A token was collected if it appeared in the top 12 tokens at the most
				activated token given the list of latent activated sequences.
			</div>
			<div className='latent-top-token-frequencies-lists'>
				<div className='latent-top-token-frequencies-list-container'>
					<div className='subtitle-2'>Top Output Tokens</div>
					<div className='latent-top-token-frequencies-list'>
						{topOutputTokenFrequencies?.length === 0 ? (
							<div className='latent-top-token-frequencies-list-loading-circle-container'>
								<LoadingCircle center={true} size='s' />
							</div>
						) : (
							<>
								{topOutputTokenFrequencies?.slice(0, 4 * 12)?.map((topOutputTokenItem, index) => (
									<div key={index} className='latent-top-token-frequencies-item'>
										<div className='latent-top-token-frequencies-item-token'>
											<i className='fa-solid fa-quote-left' />
											<span>{topOutputTokenItem?.decoded_token}</span>
											<i className='fa-solid fa-quote-right' />
										</div>
										<div className='latent-top-token-frequencies-item-frequency'>
											<span className='latent-top-token-frequencies-item-frequency-f'>f</span>
											<span>{topOutputTokenItem?.frequency}</span>
										</div>
									</div>
								))}
							</>
						)}
					</div>
				</div>
				<div className='latent-top-token-frequencies-list-container'>
					<div className='subtitle-2'>Top Layer Unembedding Tokens</div>
					<div className='latent-top-token-frequencies-list'>
						{topLayerUnembedTokenFrequencies?.length === 0 ? (
							<div className='latent-top-token-frequencies-list-loading-circle-container'>
								<LoadingCircle center={true} size='s' />
							</div>
						) : (
							<>
								{topLayerUnembedTokenFrequencies?.slice(0, 4 * 12)?.map((topOutputTokenItem, index) => (
									<div key={index} className='latent-top-token-frequencies-item'>
										<div className='latent-top-token-frequencies-item-token'>
											<i className='fa-solid fa-quote-left' />
											<span>{topOutputTokenItem?.decoded_token}</span>
											<i className='fa-solid fa-quote-right' />
										</div>
										<div className='latent-top-token-frequencies-item-frequency'>
											<span className='latent-top-token-frequencies-item-frequency-f'>f</span>
											<span>{topOutputTokenItem?.frequency}</span>
										</div>
									</div>
								))}
							</>
						)}
					</div>
				</div>
			</div>
		</div>
	);
};
