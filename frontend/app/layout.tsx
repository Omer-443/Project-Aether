import { Sidebar } from '@/components/Sidebar'
import { TopBar } from '@/components/TopBar'
import { MainGrid } from '@/components/MainGrid'
import './globals.css'

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark bg-background">
      <head>
        <title>Project Aether | Enterprise Telemetry</title>
        <meta
          name="description"
          content="Dark telemetry analytics platform for real-time monitoring and trajectory analysis"
        />
        <meta name="color-scheme" content="dark" />
        <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0d0d0d" />
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
      </head>
      <body className="font-sans antialiased">
        <Sidebar />
        <TopBar />
        <MainGrid>{children}</MainGrid>
      </body>
    </html>
  )
}
