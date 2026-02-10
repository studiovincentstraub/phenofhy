import * as vue2 from "vue";
import { MaybeRefOrGetter } from "vue";
import nprogress, { NProgressOptions } from "nprogress";

//#region useNProgress/index.d.ts
type UseNProgressOptions = Partial<NProgressOptions>;
/**
 * Reactive progress bar.
 *
 * @see https://vueuse.org/useNProgress
 */
declare function useNProgress(currentProgress?: MaybeRefOrGetter<number | null | undefined>, options?: UseNProgressOptions): {
  isLoading: vue2.WritableComputedRef<boolean, boolean>;
  progress: vue2.Ref<number | null | undefined, number | null | undefined>;
  start: () => nprogress.NProgress;
  done: (force?: boolean) => nprogress.NProgress;
  remove: () => void;
};
type UseNProgressReturn = ReturnType<typeof useNProgress>;
//#endregion
export { UseNProgressReturn as n, useNProgress as r, UseNProgressOptions as t };