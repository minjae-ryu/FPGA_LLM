//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/03/01 17:43:20
// Design Name: 
// Module Name: fp16_mul
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fp16_mul #
( 
    parameter integer OUT_W = 44
)
(
    input               clk,
    input               rst_n,
    input [2:0]         ce,
    input [16-1:0]      in_a,in_b,
    output signed [OUT_W-1:0]      o
    );


localparam integer F = OUT_W - 17;      // fractional bits
localparam integer Z = F - 6;      



wire sign_a = in_a[15];
wire sign_b = in_b[15];

wire sign_ab = sign_a^sign_b;

wire [4:0] exp_a = in_a[14:10];
wire [4:0] exp_b = in_b[14:10];

wire [9:0] mant_a = in_a[9:0];
wire [9:0] mant_b = in_b[9:0];

wire exp_a_z = ~|exp_a;
wire exp_b_z = ~|exp_b; 

wire [10:0] mant_ea = (exp_a_z) ? {1'b0,mant_a} : {1'b1,mant_a}; 
wire [10:0] mant_eb = (exp_b_z) ? {1'b0,mant_b} : {1'b1,mant_b}; 

wire [29:0] dsp_a = {19'd0,mant_ea};
wire [17:0] dsp_b = {7'd0,mant_eb};



wire zero_d = (exp_a_z&~|(mant_a)) || (exp_b_z&~|(mant_b));

//flag 고민

wire [3:0] ALUMODE = (sign_ab) ? 4'b0011 : 4'b0000;
reg [5:0] exp_sum;
reg zero0;
//st0 - dsp a,b, ALU pipeline,shift_val

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        exp_sum <=0;
        zero0 <= 0;
    end else if(ce[0])begin
        exp_sum <= {1'd0,exp_a}+{1'd0,exp_b}+{5'd0,exp_a_z}+{5'd0,exp_b_z}; 
        zero0 <= zero_d;
    end
end

//st1 - detect zero, shift, exptoshift

reg overflow,zero1;
reg [5:0] shift_val;
wire hard_overflow = (exp_sum >= 6'd47);
reg s_45;
reg s_46;
wire max_exp = (exp_sum>=46) ? 1'b1:1'b0;


always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        overflow  <= 1'b0;
        zero1     <= 1'b0;
        shift_val <= 6'd0;
        s_45      <= 1'b0;
        s_46      <= 1'b0;
    end else if (ce[1])begin
        overflow  <= hard_overflow;
        zero1     <= zero0;
        shift_val <= (exp_sum <= 6'd44) ? (6'd44 - exp_sum) : 6'd0;
        s_45      <= (exp_sum == 6'd45);
        s_46      <= (exp_sum == 6'd46);
    end
end

wire [47:0] p;

dsp dsp_uut(
    .clk(clk),
    .rst(~rst_n),
    .ALUMODE(ALUMODE),
    .ce(ce[1:0]),
    .a(dsp_a),
    .b(dsp_b),
    .p(p)
);

wire s45_overflow = p[22] ^ p[21];
wire s46_overflow = |(p[22:20] ^ {3{p[22]}});



wire signed [OUT_W-1:0] dsp_val =
    (s_45 && !s45_overflow) ? $signed({p[21:0], {(Z+1){1'b0}}}) :  // F-5 zeros  (exp_sum=45)
    (s_46 && !s46_overflow) ? $signed({p[20:0], {(Z+2){1'b0}}}) :  // F-4 zeros  (exp_sum=46)
                             $signed({p[22:0], { Z   {1'b0}}});    // F-6 zeros  (normal)

wire signed [OUT_W-1:0] shift_result = dsp_val >>> shift_val;

wire signed [OUT_W-1:0] overflow_result =
    p[47] ? $signed({1'b1, {(OUT_W-1){1'b0}}}) :  
           $signed({1'b0, {(OUT_W-1){1'b1}}});    

reg signed [OUT_W-1:0] o_shift;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        o_shift <= 1'b0;
    end else if(ce[2]) begin
        if(zero1) begin
            o_shift <= 1'b0;
        end else if(s_45) begin
            o_shift <= s45_overflow ? overflow_result : shift_result;
        end else if(s_46) begin
            o_shift <= s46_overflow ? overflow_result : shift_result;
        end else if(overflow) begin
            o_shift <= overflow_result;
        end else begin
            o_shift <= shift_result;
        end
    end
end

assign o = o_shift;

endmodule


module dsp(
    input clk,
    input rst,
    input [3:0] ALUMODE,
    input [1:0] ce,
    input [30-1:0] a,
    input [18-1:0] b,

    output [48-1:0] p
);

localparam OPMODE = 7'b0000101;
localparam INMODE = 5'b00000; // A2 B2 sel




//  <-----Cut code below this line---->

// Selects the number of A input registers. When 1 is selected, the A2 register is used. 
// Selects the number of B input registers. When 1 is selected, the B2 register is used. 




   // DSP48E1: 48-bit Multi-Functional Arithmetic Block
   //          Artix-7
   // Xilinx HDL Language Template, version 2024.2

   DSP48E1 #(
      // Feature Control Attributes: Data Path Selection
      .A_INPUT("DIRECT"),               // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
      .B_INPUT("DIRECT"),               // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
      .USE_DPORT("FALSE"),              // Select D port usage (TRUE or FALSE)
      .USE_MULT("MULTIPLY"),            // Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
      .USE_SIMD("ONE48"),               // SIMD selection ("ONE48", "TWO24", "FOUR12")
      // Pattern Detector Attributes: Pattern Detection Configuration
      .AUTORESET_PATDET("NO_RESET"),    // "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH" 
      .MASK(48'h3fffffffffff),          // 48-bit mask value for pattern detect (1=ignore)
      .PATTERN(48'h000000000000),       // 48-bit pattern match for pattern detect
      .SEL_MASK("MASK"),                // "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2" 
      .SEL_PATTERN("PATTERN"),          // Select pattern value ("PATTERN" or "C")
      .USE_PATTERN_DETECT("NO_PATDET"), // Enable pattern detect ("PATDET" or "NO_PATDET")
      // Register Control Attributes: Pipeline Register Configuration
      .ACASCREG(1),                     // Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
      .ADREG(1),                        // Number of pipeline stages for pre-adder (0 or 1)
      .ALUMODEREG(1),                   // Number of pipeline stages for ALUMODE (0 or 1)
      .AREG(1),                         // Number of pipeline stages for A (0, 1 or 2)
      .BCASCREG(1),                     // Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
      .BREG(1),                         // Number of pipeline stages for B (0, 1 or 2)
      .CARRYINREG(0),                   // Number of pipeline stages for CARRYIN (0 or 1)
      .CARRYINSELREG(0),                // Number of pipeline stages for CARRYINSEL (0 or 1)
      .CREG(0),                         // Number of pipeline stages for C (0 or 1)
      .DREG(0),                         // Number of pipeline stages for D (0 or 1)
      .INMODEREG(0),                    // Number of pipeline stages for INMODE (0 or 1)
      .MREG(0),                         // Number of multiplier pipeline stages (0 or 1)
      .OPMODEREG(0),                    // Number of pipeline stages for OPMODE (0 or 1)
      .PREG(1)                          // Number of pipeline stages for P (0 or 1)
   )
   DSP48E1_inst (
      // Cascade: 30-bit (each) output: Cascade Ports
      .ACOUT(),                   // 30-bit output: A port cascade output
      .BCOUT(),                   // 18-bit output: B port cascade output
      .CARRYCASCOUT(),     // 1-bit output: Cascade carry output
      .MULTSIGNOUT(),       // 1-bit output: Multiplier sign cascade output
      .PCOUT(),                   // 48-bit output: Cascade output
      // Control: 1-bit (each) output: Control Inputs/Status Bits
      .OVERFLOW(),             // 1-bit output: Overflow in add/acc output
      .PATTERNBDETECT(), // 1-bit output: Pattern bar detect output
      .PATTERNDETECT(),   // 1-bit output: Pattern detect output
      .UNDERFLOW(),           // 1-bit output: Underflow in add/acc output
      // Data: 4-bit (each) output: Data Ports
      .CARRYOUT(),             // 4-bit output: Carry output
      .P(p),                           // 48-bit output: Primary data output
      // Cascade: 30-bit (each) input: Cascade Ports
      .ACIN(30'd0),                     // 30-bit input: A cascade data input
      .BCIN(18'd0),                     // 18-bit input: B cascade input
      .CARRYCASCIN(1'b0),       // 1-bit input: Cascade carry input
      .MULTSIGNIN(1'b0),         // 1-bit input: Multiplier sign input
      .PCIN(48'd0),                     // 48-bit input: P cascade input
      // Control: 4-bit (each) input: Control Inputs/Status Bits
      .ALUMODE(ALUMODE),               // 4-bit input: ALU control input
      .CARRYINSEL(3'd0),         // 3-bit input: Carry select input
      .CLK(clk),                       // 1-bit input: Clock input
      .INMODE(INMODE),                 // 5-bit input: INMODE control input
      .OPMODE(OPMODE),                 // 7-bit input: Operation mode input
      // Data: 30-bit (each) input: Data Ports
      .A(a),                           // 30-bit input: A data input
      .B(b),                           // 18-bit input: B data input
      .C(48'd0),                       // 48-bit input: C data input
      .CARRYIN(1'd0),                  // 1-bit input: Carry input signal
      .D(25'd0),                       // 25-bit input: D data input
      // Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
      .CEA1(1'b0),                     // 1-bit input: Clock enable input for 1st stage AREG
      .CEA2(ce[0]),                    // 1-bit input: Clock enable input for 2nd stage AREG
      .CEAD(1'b0),                     // 1-bit input: Clock enable input for ADREG
      .CEALUMODE(ce[0]),           // 1-bit input: Clock enable input for ALUMODE
      .CEB1(1'b0),                     // 1-bit input: Clock enable input for 1st stage BREG
      .CEB2(ce[0]),                    // 1-bit input: Clock enable input for 2nd stage BREG
      .CEC(1'b0),                      // 1-bit input: Clock enable input for CREG
      .CECARRYIN(1'b0),                // 1-bit input: Clock enable input for CARRYINREG
      .CECTRL(1'b0),                   // 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
      .CED(1'b0),                      // 1-bit input: Clock enable input for DREG
      .CEINMODE(1'b0),                 // 1-bit input: Clock enable input for INMODEREG
      .CEM(1'b0),                       // 1-bit input: Clock enable input for MREG
      .CEP(ce[1]),                       // 1-bit input: Clock enable input for PREG
      .RSTA(rst),                     // 1-bit input: Reset input for AREG
      .RSTALLCARRYIN(rst),   // 1-bit input: Reset input for CARRYINREG
      .RSTALUMODE(rst),         // 1-bit input: Reset input for ALUMODEREG
      .RSTB(rst),                     // 1-bit input: Reset input for BREG
      .RSTC(rst),                     // 1-bit input: Reset input for CREG
      .RSTCTRL(rst),               // 1-bit input: Reset input for OPMODEREG and CARRYINSELREG
      .RSTD(rst),                     // 1-bit input: Reset input for DREG and ADREG
      .RSTINMODE(rst),           // 1-bit input: Reset input for INMODEREG
      .RSTM(rst),                     // 1-bit input: Reset input for MREG
      .RSTP(rst)                      // 1-bit input: Reset input for PREG
   );

   // End of DSP48E1_inst instantiation
				
			


endmodule
