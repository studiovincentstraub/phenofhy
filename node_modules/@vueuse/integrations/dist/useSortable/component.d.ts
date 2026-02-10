import { n as UseSortableReturn, t as UseSortableOptions } from "../index-CDFJRy7j.js";
import * as vue7 from "vue";
import { Reactive, SlotsType } from "vue";
import { RenderableComponent } from "@vueuse/core";

//#region useSortable/component.d.ts
interface UseSortableProps extends RenderableComponent {
  modelValue: any[];
  options?: UseSortableOptions;
}
interface UseSortableSlots {
  default: (data: Reactive<UseSortableReturn>) => any;
}
declare const UseSortable: vue7.DefineSetupFnComponent<UseSortableProps, Record<string, never>, SlotsType<UseSortableSlots>, UseSortableProps & {
  [x: `on${Capitalize<string>}`]: ((...args: unknown[]) => any) | undefined;
}, vue7.PublicProps>;
//#endregion
export { UseSortable, UseSortableProps };