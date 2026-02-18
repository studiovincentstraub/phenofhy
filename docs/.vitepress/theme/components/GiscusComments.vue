<template>
  <div ref="container"></div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

const props = withDefaults(
  defineProps<{
    repo: string
    repoId: string
    category: string
    categoryId: string
    mapping?: string
    term?: string
    strict?: string
    reactionsEnabled?: string
    emitMetadata?: string
    inputPosition?: string
    theme?: string
    lang?: string
  }>(),
  {
    mapping: 'specific',
    term: 'community-forum',
    strict: '0',
    reactionsEnabled: '1',
    emitMetadata: '0',
    inputPosition: 'bottom',
    theme: 'preferred_color_scheme',
    lang: 'en'
  }
)

const container = ref<HTMLElement | null>(null)

onMounted(() => {
  if (!container.value) return

  container.value.innerHTML = ''

  const script = document.createElement('script')
  script.src = 'https://giscus.app/client.js'
  script.async = true
  script.crossOrigin = 'anonymous'

  script.setAttribute('data-repo', props.repo)
  script.setAttribute('data-repo-id', props.repoId)
  script.setAttribute('data-category', props.category)
  script.setAttribute('data-category-id', props.categoryId)
  script.setAttribute('data-mapping', props.mapping)
  script.setAttribute('data-term', props.term)
  script.setAttribute('data-strict', props.strict)
  script.setAttribute('data-reactions-enabled', props.reactionsEnabled)
  script.setAttribute('data-emit-metadata', props.emitMetadata)
  script.setAttribute('data-input-position', props.inputPosition)
  script.setAttribute('data-theme', props.theme)
  script.setAttribute('data-lang', props.lang)

  container.value.appendChild(script)
})
</script>
