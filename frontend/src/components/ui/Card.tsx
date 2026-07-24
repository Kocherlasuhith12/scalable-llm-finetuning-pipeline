"use client";

import React from "react";
import { motion } from "framer-motion";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
  onClick?: () => void;
}

export const Card: React.FC<CardProps> = ({ children, className = "", hoverable = true, onClick }) => {
  return (
    <motion.div
      whileHover={hoverable ? { y: -3 } : undefined}
      transition={{ duration: 0.2, ease: "easeOut" }}
      onClick={onClick}
      className={`bg-[#171A21]/90 backdrop-blur-md border border-[#27272A] rounded-xl p-6 shadow-lg transition-all ${
        hoverable
          ? "hover:border-[#8B5CF6]/50 hover:shadow-xl hover:shadow-[#8B5CF6]/10"
          : ""
      } ${onClick ? "cursor-pointer" : ""} ${className}`}
    >
      {children}
    </motion.div>
  );
};

export const CardHeader: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
  <div className={`flex items-center justify-between border-b border-[#27272A]/80 pb-4 mb-4 ${className}`}>
    {children}
  </div>
);

export const CardTitle: React.FC<{ children: React.ReactNode; icon?: React.ReactNode; className?: string }> = ({
  children,
  icon,
  className = ""
}) => (
  <h3 className={`text-sm font-bold text-[#F8FAFC] flex items-center gap-2 tracking-tight ${className}`}>
    {icon && <span className="text-[#8B5CF6] flex-shrink-0">{icon}</span>}
    {children}
  </h3>
);

export const CardDescription: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ""
}) => <p className={`text-xs text-[#94A3B8] leading-relaxed mt-0.5 ${className}`}>{children}</p>;

export const CardContent: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ""
}) => <div className={`space-y-4 ${className}`}>{children}</div>;

export const CardFooter: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className = ""
}) => <div className={`pt-4 border-t border-[#27272A]/80 flex items-center justify-between ${className}`}>{children}</div>;
