import type { Metadata } from "next"
import "./globals.css"
import { AppShell } from "@/components/app-shell"
import { Toaster } from "@/components/ui/toaster"

export const metadata: Metadata = {
  title: "易图 — 图数据智能分析平台",
  description:
    "用自然语言提问，ReAct 智能体逐轮调用图算法、观测结果并生成分析报告。",
  icons: {
    icon: "/icon.svg",
    apple: "/apple-icon.png",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" className="bg-background">
      <body
        className="font-sans antialiased"
      >
        <AppShell>{children}</AppShell>
        <Toaster />
      </body>
    </html>
  )
}
