import { isRef, nextTick, toValue } from "vue";
import { defaultDocument, tryOnMounted, tryOnScopeDispose, unrefElement } from "@vueuse/core";
import Sortable from "sortablejs";

//#region useSortable/index.ts
/**
* Wrapper for sortablejs.
* @param el
* @param list
* @param options
*/
function useSortable(el, list, options = {}) {
	let sortable;
	const { document = defaultDocument,...resetOptions } = options;
	const defaultOptions = { onUpdate: (e) => {
		moveArrayElement(list, e.oldIndex, e.newIndex, e);
	} };
	const start = () => {
		const target = typeof el === "string" ? document === null || document === void 0 ? void 0 : document.querySelector(el) : unrefElement(el);
		if (!target || sortable !== void 0) return;
		sortable = new Sortable(target, {
			...defaultOptions,
			...resetOptions
		});
	};
	const stop = () => {
		sortable === null || sortable === void 0 || sortable.destroy();
		sortable = void 0;
	};
	const option = (name, value) => {
		if (value !== void 0) sortable === null || sortable === void 0 || sortable.option(name, value);
		else return sortable === null || sortable === void 0 ? void 0 : sortable.option(name);
	};
	tryOnMounted(start);
	tryOnScopeDispose(stop);
	return {
		stop,
		start,
		option
	};
}
/**
* Inserts a element into the DOM at a given index.
* @param parentElement
* @param element
* @param {number} index
* @see https://github.com/Alfred-Skyblue/vue-draggable-plus/blob/a3829222095e1949bf2c9a20979d7b5930e66f14/src/utils/index.ts#L81C1-L94C2
*/
function insertNodeAt(parentElement, element, index) {
	const refElement = parentElement.children[index];
	parentElement.insertBefore(element, refElement);
}
/**
* Removes a node from the DOM.
* @param {Node} node
* @see https://github.com/Alfred-Skyblue/vue-draggable-plus/blob/a3829222095e1949bf2c9a20979d7b5930e66f14/src/utils/index.ts#L96C1-L102C2
*/
function removeNode(node) {
	if (node.parentNode) node.parentNode.removeChild(node);
}
function moveArrayElement(list, from, to, e = null) {
	if (e != null) {
		removeNode(e.item);
		insertNodeAt(e.from, e.item, from);
	}
	const _valueIsRef = isRef(list);
	const array = _valueIsRef ? [...toValue(list)] : toValue(list);
	if (to >= 0 && to < array.length) {
		const element = array.splice(from, 1)[0];
		nextTick(() => {
			array.splice(to, 0, element);
			if (_valueIsRef) list.value = array;
		});
	}
}

//#endregion
export { useSortable as i, moveArrayElement as n, removeNode as r, insertNodeAt as t };