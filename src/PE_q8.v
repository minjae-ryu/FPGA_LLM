module PE_q8 (
    input clk,
    input rst_n,
    
    input i_valid,
    input a_valid,

    input [8*64-1:0] w,
    input [32-1:0]   ws,
    input [8*64-1:0] a,
    input [32-1:0]   as,


    output [32-1:0] o,    
    output o_valid,

);


reg [7-1:0] delay;
reg [7-1:0] a_delay;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        delay <= 0;
        a_delay <= 0;
    end else begin
        delay <= {delay[7-2:0],i_valid};
        a_delay <= {a_delay[7-2:0],i_valid};
    end
end



wire signed [8-1:0] w_vec[0:64-1];
wire signed [8-1:0] a_vec[0:64-1];
reg signed [16-1:0] mul_vec[0:64-1];

genvar k;

generate
    for(k = 0; k < 64; k = k + 1)begin
        assign w_vec[k] = w[(k+1)*8-1:k*8];
        assign a_vec[k] = a[(k+1)*8-1:k*8];
        always @(posedge clk) begin
            if(i_valid)begin
                mul_vec[k] <= w_vec[k]*a_vec[k];
            end
        end
    end
endgenerate


reg signed [17-1:0] st0_vec[0:32-1];
reg signed [18-1:0] st1_vec[0:16-1];
reg signed [19-1:0] st2_vec[0:8-1];
reg signed [20-1:0] st3_vec[0:4-1];
reg signed [21-1:0] st4_vec[0:2-1];
reg signed [22-1:0] st5_vec;

always @(posedge clk) begin
    if (delay[0]) begin
        for (integer i = 0; i < 32; i = i + 1) begin
            st0_reg[i] <= mul_vec[2*i] + mul_vec[2*i+1];
        end
    end
end

always @(posedge clk) begin
    if (delay[1]) begin
        for (integer i = 0; i < 16; i = i + 1) begin
            st1_reg[i] <= st0_reg[2*i] + st0_reg[2*i+1];
        end
    end
end

always @(posedge clk) begin
    if (delay[2]) begin
        for (integer i = 0; i < 8; i = i + 1) begin
            st2_reg[i] <= st1_reg[2*i] + st1_reg[2*i+1];
        end
    end
end

always @(posedge clk) begin
    if (delay[3]) begin
        for (integer i = 0; i < 4; i = i + 1) begin
            st3_reg[i] <= st2_reg[2*i] + st2_reg[2*i+1];
        end
    end
end

always @(posedge clk) begin
    if (delay[4]) begin
        for (integer i = 0; i < 2; i = i + 1) begin
            st4_reg[i] <= st3_reg[2*i] + st3_reg[2*i+1];
        end
    end
end


always @(posedge clk) begin
    if (delay[5]) begin
        st5_reg <= st4_reg[0] + st4_reg[1];
    end
end








endmodule