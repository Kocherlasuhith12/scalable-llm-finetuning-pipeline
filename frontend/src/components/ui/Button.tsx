"use client";

import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

export interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  children?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  children,
  className = "",
  disabled,
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center justify-center font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-[#E11D48]/50 disabled:opacity-50 disabled:cursor-not-allowed select-none cursor-pointer";

  const variants = {
    primary:
      "bg-gradient-to-r from-[#E11D48] to-[#9F1239] text-[#F8FAFC] shadow-md shadow-[#E11D48]/25 hover:from-[#F43F5E] hover:to-[#E11D48] hover:shadow-[#E11D48]/40 active:scale-[0.98]",
    secondary:
      "bg-[#18181F] text-[#F8FAFC] border border-[#2A2A35] hover:bg-[#22222B] hover:border-[#3f3f4e] hover:text-[#F8FAFC]",
    outline:
      "bg-transparent text-[#F43F5E] border border-[#E11D48]/40 hover:bg-[#E11D48]/10 hover:border-[#E11D48]",
    ghost:
      "bg-transparent text-[#94A3B8] hover:bg-[#18181F] hover:text-[#F8FAFC]",
    danger:
      "bg-[#EF4444]/10 text-[#EF4444] border border-[#EF4444]/30 hover:bg-[#EF4444]/20 hover:border-[#EF4444]/50"
  };

  const sizes = {
    sm: "px-3 py-1.5 text-xs gap-1.5",
    md: "px-4 py-2 text-xs font-semibold gap-2",
    lg: "px-5 py-2.5 text-sm font-semibold gap-2.5"
  };

  return (
    <motion.button
      whileTap={{ scale: disabled || isLoading ? 1 : 0.98 }}
      whileHover={{ y: disabled || isLoading ? 0 : -1 }}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <svg className="animate-spin w-3.5 h-3.5 text-current" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
      ) : (
        leftIcon
      )}
      {children && <span>{children}</span>}
      {!isLoading && rightIcon}
    </motion.button>
  );
};
