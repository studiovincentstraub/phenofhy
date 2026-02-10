import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Phenofhy",
  description: "Python package for phenotype data processing and analysis within the Our Future Health TRE",
  base: '/phenofhy/',

  // preferred (maintainer-recommended) way to set initial appearance
  appearance: {
    // @ts-expect-error not fully supported in the types yet
    initialValue: 'light'
  },

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started/quickstart' },
      { text: 'Concepts', link: '/concepts/overview' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Support', link: '/support/faq' },
    ],
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Environment & Setup', link: '/getting-started/environment' },
          { text: 'Installation', link: '/getting-started/installation' },
          { text: 'Quickstart', link: '/getting-started/quickstart' },
        ]
      },
      {
        text: 'Concepts',
        items: [
          { text: 'Overview', link: '/concepts/overview' },
          { text: 'Data Model', link: '/concepts/data-model' },
          { text: 'Workflow', link: '/concepts/pipeline' },
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Overview', link: '/api/' },
          { text: 'pipeline', link: '/api/pipeline' },
          { text: 'load', link: '/api/load' },
          { text: 'extract', link: '/api/extract' },
          { text: 'process', link: '/api/process' },
          { text: 'calculate', link: '/api/calculate' },
          { text: 'profile', link: '/api/profile' },
          { text: 'icd', link: '/api/icd' },
          { text: 'display', link: '/api/display' },
          { text: 'utils', link: '/api/utils' },
          { text: '_rules', link: '/api/rules' },
          { text: '_derive_funcs', link: '/api/derive' },
          { text: '_filter_funcs', link: '/api/filter' },
        ]
      },
      {
        text: 'Support',
        items: [
          { text: 'FAQ', link: '/support/faq' },
          { text: 'Bug Reports', link: '/support/bug-reports' },
        ]
      }
    ]
  }
})
