import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Phenofhy",
  description: "Python package for phenotype data processing and analysis within the Our Future Health TRE",
  base: '/phenofhy/',
  head: [['link', { rel: 'icon', href: '/logo/phenofhy-icon.ico' }]],

  // disable dark mode
  appearance: false,

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Support', link: '/support/bug-reports' },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/studiovincentstraub/phenofhy' }
    ],
    footer: {
      message: 'Released under a BSD-3-Clause License.',
      copyright: 'Copyright © 2026 <a href="https://www.vincentjstraub.com/" target="_blank" rel="noreferrer">Studio Vincent Straub</a>'
    },
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
        text: 'Key concepts',
        items: [
          { text: 'Overview', link: '/concepts/overview' },
          { text: 'Data Model', link: '/concepts/data-model' },
          { text: 'Workflow', link: '/concepts/pipeline' },
        ]
      },
      {
        text: 'Tutorials',
        items: [
          { text: 'Running a pipeline', link: '/concepts/tutorials' },
          { text: 'Profiling a phenotype', link: '/tutorials/profile' },
          { text: 'Calculating prevalence', link: '/tutorials/calculate' },
          { text: 'ICD phenotypes', link: '/tutorials/icd' },
          { text: 'TRE utilities', link: '/tutorials/utilities' },
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
