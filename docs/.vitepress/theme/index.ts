import DefaultTheme from 'vitepress/theme'
import './custom.css'
import GiscusComments from './components/GiscusComments.vue'

export default {
	...DefaultTheme,
	enhanceApp({ app }) {
		app.component('GiscusComments', GiscusComments)
	}
}
