// Packages
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

// Components

// Logic

// Context

// Services

// Styles

// Assets

export const TextBoxLogic = ({ value, setValue, onEnter, focusOnStart, blurOnEnter }) => {
	const textBoxNode = useRef();
	const [textBoxWidth, setTextBoxWidth] = useState(0);
	const textBoxRef = useCallback(
		(node) => {
			textBoxNode.current = node;
			setTextBoxWidth(node?.clientWidth);
		},
		[setTextBoxWidth]
	);

	useEffect(() => {
		if (focusOnStart) textBoxNode?.current?.focus();
	}, [textBoxNode, focusOnStart]);

	const textBoxHeightBoxNode = useRef();
	const [textBoxHeightBoxHeight, setTextBoxHeightBoxHeight] = useState(0);
	const textBoxHeightBoxRef = useCallback(
		(node) => {
			textBoxHeightBoxNode.current = node;
			setTextBoxHeightBoxHeight(node?.clientHeight);
		},
		[setTextBoxHeightBoxHeight]
	);
	const updateTextBoxHeight = useCallback(() => {
		setTextBoxWidth(textBoxNode?.current?.clientWidth);
		setTextBoxHeightBoxHeight(textBoxHeightBoxNode?.current?.clientHeight);
	}, [setTextBoxHeightBoxHeight, setTextBoxWidth]);
	useLayoutEffect(() => {
		updateTextBoxHeight();
		setTimeout(() => updateTextBoxHeight(), 1);
	}, [value, updateTextBoxHeight]);

	function onKeyDown(e) {
		if (onEnter !== undefined && e?.code === "Enter" && !e?.ctrlKey) {
			e?.preventDefault();
			if (blurOnEnter) textBoxNode?.current?.blur();
			onEnter();
		} else if (e?.code === "Enter" && e?.ctrlKey) {
			const start = e.target.selectionStart;
			const end = e.target.selectionEnd;

			const newValue = value.substring(0, start) + "\n" + value.substring(end);
			setValue(newValue);

			setTimeout(() => {
				e.target.selectionStart = e.target.selectionEnd = start + 1;
			}, 0);
		}
	}

	const onTextBoxChange = useCallback(
		(e) => {
			if (e?.target?.value !== undefined) {
				setValue(e?.target?.value);
			} else if (e !== undefined) {
				setValue(e);
			}

			updateTextBoxHeight();
		},
		[setValue, updateTextBoxHeight]
	);

	return { textBoxRef, textBoxHeightBoxRef, textBoxWidth, textBoxHeightBoxHeight, onTextBoxChange, onKeyDown };
};
