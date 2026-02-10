import { n as UseFocusTrapReturn, t as UseFocusTrapOptions } from "../index-CJw_eFkJ.js";
import * as vue4 from "vue";
import { Reactive, SlotsType } from "vue";
import { RenderableComponent } from "@vueuse/core";

//#region useFocusTrap/component.d.ts
interface UseFocusTrapProps extends RenderableComponent {
  options?: UseFocusTrapOptions;
}
/**
 * @deprecated
 */
interface ComponentUseFocusTrapOptions extends UseFocusTrapProps {}
interface UseFocusTrapSlots {
  default: (data: Reactive<UseFocusTrapReturn>) => any;
}
declare const UseFocusTrap: vue4.DefineSetupFnComponent<UseFocusTrapProps, Record<string, never>, SlotsType<UseFocusTrapSlots>, UseFocusTrapProps & {
  [x: `on${Capitalize<string>}`]: ((...args: unknown[]) => any) | undefined;
}, vue4.PublicProps>;
//#endregion
export { ComponentUseFocusTrapOptions, UseFocusTrap, UseFocusTrapProps };