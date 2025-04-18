import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Réseau de Neurones Interactif',
  description: 'Outil pédagogique pour les réseaux de neurones avec apprentissage en temps réel',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fr">
      <body className={`${inter.className} bg-gray-100 min-h-screen`}>{children}</body>
    </html>
  );
}