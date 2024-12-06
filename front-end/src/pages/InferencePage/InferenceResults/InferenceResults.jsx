// Packages

// Components

// Logic
import { InferenceResultsLogic } from "./InferenceResultsLogic";

// Context

// Services

// Styles
import "./InferenceResults.css";
import { LoadingCircle } from "../../../components/LoadingCircle/LoadingCircle";

// Assets

export const InferenceResults = () => {
	const {
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
	} = InferenceResultsLogic();

	return (
		<div
			className={
				"inference-results-container" +
				(inferenceResults !== false ? " inference-results-container-show" : "") +
				(isViewingInferenceResult ? " inference-results-container-viewing-result" : "")
			}
		>
			<div className='inference-results'>
				{inferenceResults === false
					? null
					: inferenceResults?.map((inferenceResult, inferenceResultIndex) => (
							<div
								key={inferenceResultIndex}
								className='inference-results-item'
								onClick={() => onClickResultsItem(inferenceResult?.inference_id)}
							>
								<div className='inference-results-item-tokens'>
									{inferenceResult?.tokens?.map((token, tokenIndex) => (
										<span key={tokenIndex}>{token}</span>
									))}
								</div>
							</div>
					  ))}
			</div>
			{viewingInferenceResultId === false ? null : (
				<div className='inference-result-container'>
					<div className='inference-result'>
						<button className='inference-result-back-btn' onClick={backToInferenceResults}>
							<i className='fa-solid fa-chevron-left' />
							<span>Back to Inference Results</span>
						</button>
						<div className='inference-result-tokens'>
							{inferenceResults
								?.find((e) => e?.inference_id === viewingInferenceResultId)
								?.tokens?.map((token, tokenIndex) => (
									<span
										key={tokenIndex}
										onClick={() => onClickToken(tokenIndex)}
										className={sequenceOfThoughtsTokenIndex === tokenIndex ? " inference-result-tokens-token-active" : ""}
									>
										<span>{token}</span>
									</span>
								))}
						</div>
						<div className='inference-result-latents-layers-container'>
							{sequenceOfThoughtsTopLatents === false ? (
								!sequenceOfThoughtsTokenIndex ? null : (
									<div className='inference-result-latents-layers-loading-circle-container'>
										<LoadingCircle center={true} label='Gathering Latent Connections...' />
									</div>
								)
							) : (
								Array(12)
									.fill(0)
									.map((_, layerIndex) => (
										<div key={layerIndex} className='inference-result-latents-layer'>
											<div className='inference-result-latents-layer-label'>
												Layer {(layerIndex + 1).toString()?.length < 2 ? "0" : ""}
												{layerIndex + 1}
											</div>
											<div className='inference-result-latents-layer-latents'>
												{sequenceOfThoughtsTopLatents[layerIndex]?.map((topLatent, latentIndex) => (
													<div
														ref={latentsLatentRef}
														key={latentIndex}
														className='inference-result-latents-latent-container'
														onMouseDown={(e) => e?.preventDefault()}
														onMouseEnter={() => onLatentMouseEnter(layerIndex, latentIndex)}
														onMouseLeave={() => onLatentMouseLeave(layerIndex, latentIndex)}
													>
														<div className='inference-result-latents-latent'>
															<button
																onClick={(e) => goToLatent(e, layerIndex, topLatent?.latent)}
																onAuxClick={(e) => goToLatent(e, layerIndex, topLatent?.latent)}
															>
																<span>{topLatent?.latent + 1}</span>
															</button>
															<div className='inference-result-latents-latent-preview'>
																<div className='inference-result-latents-latent-preview-location'>
																	<span>Layer {layerIndex + 1}</span>
																	<span>Latent {topLatent?.latent + 1}</span>
																</div>
																<div className='inference-result-latents-latent-preview-label'>Top Sequences</div>
																<div className='inference-result-latents-latent-preview-top-sequences'>
																	{topLatent?.topSequences?.map((topSequence, topSequenceIndex) => (
																		<div
																			key={topSequenceIndex}
																			className='inference-result-latents-latent-preview-top-sequence'
																		>
																			{topSequence}
																		</div>
																	))}
																</div>
															</div>
															{topLatent?.relationships
																?.filter((e) => e[0][1][0] == layerIndex + 1)
																?.map((e) => {
																	return {
																		frequency: e[1],
																		connectedLatentPosition: sequenceOfThoughtsTopLatents[
																			layerIndex + 1
																		]?.findIndex((e2) => e2?.latent === e[0][1][1]),
																	};
																})
																?.map((relationship, index) => (
																	<div
																		key={index}
																		className={
																			"inference-result-latents-latent-connection-container" +
																			(JSON.stringify(latentPositionMouseOver) ===
																			JSON.stringify([layerIndex + 1, relationship?.connectedLatentPosition])
																				? " inference-result-latents-latent-connection-container-active"
																				: latentPositionMouseOver !== false
																				? " inference-result-latents-latent-connection-container-inactive"
																				: "")
																		}
																		style={{
																			"--intensity": relationship?.frequency / 160,
																		}}
																	>
																		<div
																			key={index}
																			className='inference-result-latents-latent-connection'
																			style={{
																				"--angle":
																					-1 *
																						(Math.atan(
																							(latentsLatentWidth *
																								relationship?.connectedLatentPosition -
																								latentsLatentWidth * latentIndex) /
																								(160 - 9 - 4)
																						) *
																							(180 / Math.PI)) +
																					"deg",
																				"--length":
																					Math.hypot(
																						latentsLatentWidth * relationship?.connectedLatentPosition -
																							latentsLatentWidth * latentIndex,
																						160 - 9 - 4
																					) + "px",
																			}}
																		></div>
																		<div
																			className='inference-result-latents-latent-connection-frequency inference-result-latents-latent-connection-frequency-before'
																			style={{
																				"--left":
																					(relationship?.connectedLatentPosition - latentIndex) *
																						(latentsLatentWidth / 4.5) +
																					"px",
																			}}
																		>
																			{(relationship?.frequency / 160).toFixed(3)}
																		</div>
																		<div
																			className='inference-result-latents-latent-connection-frequency inference-result-latents-latent-connection-frequency-after'
																			style={{
																				"--left":
																					latentsLatentWidth *
																						(relationship?.connectedLatentPosition - latentIndex) *
																						0.75 +
																					"px",
																			}}
																		>
																			{(relationship?.frequency / 160).toFixed(3)}
																		</div>
																	</div>
																))}
														</div>
													</div>
												))}
											</div>
										</div>
									))
							)}
						</div>
					</div>
				</div>
			)}
		</div>
	);
};
