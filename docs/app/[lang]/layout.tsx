import { defineI18nUI } from 'fumadocs-ui/i18n';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { Inter } from 'next/font/google';
import '../global.css';
import { i18n } from '@/lib/i18n';

const inter = Inter({
  subsets: ['latin'],
});

const { provider } = defineI18nUI(i18n, {
  translations: {
    en: { displayName: 'English' },
    es: {
      displayName: 'Español',
      search: 'Buscar',
      searchNoResult: 'Sin resultados',
      toc: 'En esta página',
      lastUpdate: 'Última actualización',
      chooseLanguage: 'Elegir idioma',
      nextPage: 'Siguiente',
      previousPage: 'Anterior',
      chooseTheme: 'Tema',
      editOnGithub: 'Editar en GitHub',
    },
  },
});

export default async function Layout({ params, children }: LayoutProps<'/[lang]'>) {
  const { lang } = await params;

  return (
    <html lang={lang} className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider i18n={provider(lang)}>{children}</RootProvider>
      </body>
    </html>
  );
}
